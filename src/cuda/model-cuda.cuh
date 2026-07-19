// Shared CUDA backend: everything except the matmul. Included by exactly one
// compute file per binary — model-cuda-f32.cu (f32 dequant dot) or
// model-cuda-i8.cu (int8 dot) — each of which defines matmul_q. The forward
// pass, the kv cache, and every non-matmul kernel (rmsnorm, rope, attention,
// elementwise) are identical across both, so they live here. This is a
// single-include unit, not a normal header: it holds definitions, and the
// including file provides matmul_q.

#ifndef MODEL_CUDA_CUH
#define MODEL_CUDA_CUH

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" {
#include "model.h"
#include "quant.h"
#include "mtp-internal.h"
}

// kvcache rows live on the device here, and so will the draft head: the CUDA
// MTP port (device-side assistant + batched verify + async overlap) is the
// in-progress follow-up — until it lands, drafting on this backend declines.
extern "C" const int model_kv_host = 0;

// The device draft is implemented at the end of this file (it needs d_attn,
// the norm/geglu/argmax kernels and g_hidden, all defined below).

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    exit(1); } } while (0)

static inline int gridn(int n) { return (n + 255) / 256; }

// ---- quant block layouts + fp16 helpers (used by the matmul in the .cu) ----

#define QK_K 256
typedef unsigned short ggml_half;

typedef struct { ggml_half d, dmin; uint8_t scales[12]; uint8_t qs[QK_K/2]; } block_q4_K;
typedef struct { uint8_t hmask[QK_K/8]; uint8_t qs[QK_K/4]; uint8_t scales[12]; ggml_half d; } block_q3_K;
typedef struct { ggml_half d, dmin; uint8_t scales[12]; uint8_t qh[QK_K/8]; uint8_t qs[QK_K/2]; } block_q5_K;
typedef struct { uint8_t ql[QK_K/2]; uint8_t qh[QK_K/4]; int8_t scales[QK_K/16]; ggml_half d; } block_q6_K;
typedef struct { ggml_half d; int8_t qs[32]; } block_q8_0;
typedef struct { ggml_half d; uint8_t qs[16]; } block_q4_0;   // elem j low nibble, j+16 high

__device__ static float d_fp16(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16, exp = (h >> 10) & 0x1Fu, mant = h & 0x3FFu, bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else { exp = 127 - 15 + 1; while ((mant & 0x400u) == 0) { mant <<= 1; exp--; } mant &= 0x3FFu; bits = sign | (exp << 23) | (mant << 13); }
    } else if (exp == 0x1F) bits = sign | 0x7F800000u | (mant << 13);
    else bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    float f; memcpy(&f, &bits, 4); return f;
}
__device__ static float d_bf16(uint16_t h) { uint32_t b = (uint32_t)h << 16; float f; memcpy(&f, &b, 4); return f; }

// q4_K / q5_K 6-bit sub-block scale+min unpack.
__device__ static void d_gsm(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4); }
}

// ---- activation quantization epilogue --------------------------------------
// The int8 backend wants every matmul input quantized (per 32-element group: a
// scale, the int8 values, and their sum). Rather than launching a separate
// quantize kernel per matmul (~300 launches/token re-reading vectors that some
// kernel just wrote), each activation is quantized where it is BORN: the kernels
// below take a `struct actq` and, when its pointers are set, quantize their own
// output as an epilogue. The f32 backend passes AQ0 (all NULL) and the branch
// vanishes; the int8 backend hands out its buffers via actq_for (seam, below).

// Per 32-element group: the int8 values, plus scale and value-sum packed as one
// float2 — a single load serves both (the sum is ≤ 32·127, exact as a float).
struct actq { int8_t *xq; float2 *xds; };
static const struct actq AQ0 = { NULL, NULL };

// Quantize one 32-element group g (global index) of x into int8 + (scale, sum).
__device__ static void d_quant_group(const float *xb, int g, struct actq aq) {
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(xb[i]));
    float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
    int8_t *q = aq.xq + (size_t)g * 32;
    int sum = 0;
    for (int i = 0; i < 32; i++) { int v = __float2int_rn(xb[i] * id); q[i] = (int8_t)v; sum += v; }
    aq.xds[g] = make_float2(d, (float)sum);
}

// Warp-cooperative twin for the WIDE (prefill) epilogues: lane-per-element, so
// a warp's loads coalesce into one line per step — the serial form walks 32
// consecutive floats per THREAD, touching 32 different 128-byte lines per step
// (12.5% sector efficiency) and idling 31/32 threads. fmax and the int sum are
// exactly associative, so the tree reductions produce byte-identical output.
// Decode's captured kernels keep the serial form (frozen-decode rule).
__device__ static void d_quant_group_warp(const float *xb, int g, struct actq aq, int lane) {
    float v = xb[lane], amax = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
    int q = __float2int_rn(v * id);
    aq.xq[(size_t)g * 32 + lane] = (int8_t)q;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    if (lane == 0) aq.xds[g] = make_float2(d, (float)q);
}

// ====================  non-matmul compute kernels  =========================

// RMSNorm over `rows` vectors of length n. w (length n, shared across rows) or NULL.
// One block per row — 1024 threads for the lone full-width row (a 256-thread
// block leaves the SM mostly idle and every other SM entirely idle), 256 for
// the small per-head rows.
#define NORM_THREADS(n) ((n) >= 1024 ? 1024 : 256)
__global__ static void rmsnorm_kernel(float *out, const float *x, const float *w, int n, float eps, struct actq aq) {
    int row = blockIdx.x;
    const float *xr = x + (size_t)row * n;
    float *outr = out + (size_t)row * n;
    __shared__ float sh[1024];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    sh[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s]; __syncthreads(); }
    float scale = rsqrtf(sh[0] / (float)n + eps);
    if (w) for (int i = threadIdx.x; i < n; i += blockDim.x) outr[i] = xr[i] * scale * w[i];
    else   for (int i = threadIdx.x; i < n; i += blockDim.x) outr[i] = xr[i] * scale;
    if (aq.xq) {
        __syncthreads();
        for (int g = threadIdx.x; g < n / 32; g += blockDim.x)
            d_quant_group(outr + g * 32, row * (n / 32) + g, aq);
    }
}

// The tail of every sub-layer is "rmsnorm the output, add it to the residual,
// maybe scale" — three round-trips through global memory as separate kernels.
// Fused: one block normalizes x (length n), accumulates into acc, applies the
// optional per-layer output scale os, and (epilogue) quantizes the result.
// The body is shared with the chunk form below via `row`; this decode entry
// pins row to 0 so the captured graph keeps the original instruction stream.
__device__ static void d_rmsnorm_add(float *acc, const float *x, const float *w, int n, float eps,
                                     const float *os, struct actq aq, int row) {
    const float *xr = x + (size_t)row * n;
    float *accr = acc + (size_t)row * n;
    __shared__ float sh[1024];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    sh[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s]; __syncthreads(); }
    float scale = rsqrtf(sh[0] / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        // __fadd_rn keeps the separately-rounded mul-then-add of the unfused
        // kernels (an FMA here would change the last bit, and the greedy path).
        float v = __fadd_rn(accr[i], xr[i] * scale * w[i]);
        accr[i] = os ? v * os[0] : v;
    }
    if (aq.xq) {
        __syncthreads();
        for (int g = threadIdx.x; g < n / 32; g += blockDim.x)
            d_quant_group(accr + g * 32, row * (n / 32) + g, aq);
    }
}
__global__ static void rmsnorm_add_kernel(float *acc, const float *x, const float *w, int n, float eps,
                                          const float *os, struct actq aq) {
    d_rmsnorm_add(acc, x, w, n, eps, os, aq, 0);
}
__global__ static void rmsnorm_add_n_kernel(float *acc, const float *x, const float *w, int n, float eps,
                                            const float *os, struct actq aq) {
    d_rmsnorm_add(acc, x, w, n, eps, os, aq, blockIdx.x);
}

// Warp-per-row chunk norms. The block-per-row forms above put ONE row on a
// whole SM (a 1024-thread block is alone on an Orin SM), so a 960-row chunk
// runs 120 latency-bound waves — measured 24% SM / 28% memory. A warp per row
// (8 rows per 256-thread block) keeps every scheduler fed and the epilogue
// quantizes warp-cooperatively. The shuffle-tree row sum reassociates the
// reduction (different rounding than the block tree) — a numerics-gated
// change under the f16-KV-precedent battery; decode keeps the block forms,
// frozen in its captured graph.
__device__ static float d_warp_rowsum2(const float *xr, int n, int lane) {
    float ss = 0.0f;
    if ((n & 127) == 0) {                          // float4 fast path: 4x fewer load transactions
        for (int i = lane * 4; i < n; i += 32 * 4) {
            float4 x4 = *(const float4 *)(xr + i);
            ss += x4.x * x4.x + x4.y * x4.y + x4.z * x4.z + x4.w * x4.w;
        }
    } else {
        for (int i = lane; i < n; i += 32) ss += xr[i] * xr[i];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    return ss;
}
__global__ static void rmsnorm_w_n_kernel(float *out, const float *x, const float *w, int n, float eps,
                                          struct actq aq, int rows) {
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= rows) return;
    const float *xr = x + (size_t)row * n;
    float *outr = out + (size_t)row * n;
    float scale = rsqrtf(d_warp_rowsum2(xr, n, lane) / (float)n + eps);
    if ((n & 127) == 0) {                          // float4 fast path (byte-identical multiply order)
        if (w) for (int i = lane * 4; i < n; i += 32 * 4) {
            float4 x4 = *(const float4 *)(xr + i);
            float4 w4 = *(const float4 *)(w + i);
            float4 o4;
            o4.x = x4.x * scale * w4.x;
            o4.y = x4.y * scale * w4.y;
            o4.z = x4.z * scale * w4.z;
            o4.w = x4.w * scale * w4.w;
            *(float4 *)(outr + i) = o4;
        }
        else   for (int i = lane * 4; i < n; i += 32 * 4) {
            float4 x4 = *(const float4 *)(xr + i);
            float4 o4;
            o4.x = x4.x * scale;
            o4.y = x4.y * scale;
            o4.z = x4.z * scale;
            o4.w = x4.w * scale;
            *(float4 *)(outr + i) = o4;
        }
    } else {
        if (w) for (int i = lane; i < n; i += 32) outr[i] = xr[i] * scale * w[i];
        else   for (int i = lane; i < n; i += 32) outr[i] = xr[i] * scale;
    }
    if (aq.xq) {
        __syncwarp();
        for (int g = 0; g < n / 32; g++) d_quant_group_warp(outr + g * 32, row * (n / 32) + g, aq, lane);
    }
}
__global__ static void rmsnorm_add_w_n_kernel(float *acc, const float *x, const float *w, int n, float eps,
                                              const float *os, struct actq aq, int rows) {
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= rows) return;
    const float *xr = x + (size_t)row * n;
    float *accr = acc + (size_t)row * n;
    float scale = rsqrtf(d_warp_rowsum2(xr, n, lane) / (float)n + eps);
    if ((n & 127) == 0) {                          // float4 fast path (byte-identical mul/add order)
        for (int i = lane * 4; i < n; i += 32 * 4) {
            float4 x4 = *(const float4 *)(xr + i);
            float4 w4 = *(const float4 *)(w + i);
            float4 a4 = *(const float4 *)(accr + i);
            float4 o4;
            o4.x = __fadd_rn(a4.x, x4.x * scale * w4.x);
            o4.y = __fadd_rn(a4.y, x4.y * scale * w4.y);
            o4.z = __fadd_rn(a4.z, x4.z * scale * w4.z);
            o4.w = __fadd_rn(a4.w, x4.w * scale * w4.w);
            if (os) {
                float s = os[0];
                o4.x *= s; o4.y *= s; o4.z *= s; o4.w *= s;
            }
            *(float4 *)(accr + i) = o4;
        }
    } else {
        for (int i = lane; i < n; i += 32) {
            // __fadd_rn as in d_rmsnorm_add: keep the separately-rounded mul-then-add
            float v = __fadd_rn(accr[i], xr[i] * scale * w[i]);
            accr[i] = os ? v * os[0] : v;
        }
    }
    if (aq.xq) {
        __syncwarp();
        for (int g = 0; g < n / 32; g++) d_quant_group_warp(accr + g * 32, row * (n / 32) + g, aq, lane);
    }
}
// Chunk-norm dispatch. The warp forms win when rows fill the schedulers; a
// TINY chunk — above all the B=3 MTP verify — must keep the block-per-row
// forms, for two reasons: a 1-block warp-form grid serializes on one SM
// (measured +20% on the verify pass), and the verify's argmax must match
// plain decode's BY CONSTRUCTION, which requires the verify chunk to run
// decode's exact norm reduction order. B >= 16 is chunks only; every
// verify/remainder stays on the decode-identical path.
static void norm_n(float *out, const float *x, const float *w, int n, float eps, struct actq aq, int B) {
    if (B >= 16) rmsnorm_w_n_kernel<<<(B + 7) / 8, 256>>>(out, x, w, n, eps, aq, B);
    else         rmsnorm_kernel<<<B, NORM_THREADS(n)>>>(out, x, w, n, eps, aq);
}
static void norm_add_n(float *acc, const float *x, const float *w, int n, float eps,
                       const float *os, struct actq aq, int B) {
    if (B >= 16) rmsnorm_add_w_n_kernel<<<(B + 7) / 8, 256>>>(acc, x, w, n, eps, os, aq, B);
    else         rmsnorm_add_n_kernel<<<B, NORM_THREADS(n)>>>(acc, x, w, n, eps, os, aq);
}

// GPT-NeoX RoPE: `total` = n_head * (hd/2) elements; v is [n_head][hd].
__global__ static void rope_kernel(float *v, int half, int hd, const int *d_pos, float base, const float *ff, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int pos = *d_pos;                                  // device-resident: lets the forward be a static graph
    int head = idx / half, i = idx % half;
    float *vh = v + (size_t)head * hd;
    float freq = powf(base, -2.0f * (float)i / (float)hd);
    float ang = (float)pos * freq / (ff ? ff[i] : 1.0f);
    float c = cosf(ang), s = sinf(ang), a = vh[i], b = vh[i + half];
    vh[i] = a * c - b * s; vh[i + half] = a * s + b * c;
}
// The chunk form: row r holds the chunk's r-th token, at position *d_pos + r.
// A separate kernel (not a `rows` parameter) so the decode graph keeps the
// division-free original — its tiny latency-bound nodes notice every cycle.
// The chunk form takes a PRECOMPUTED frequency table: the frequency depends
// only on i — hd/2 distinct values — yet the inline form recomputed powf per
// (row, head, i), ~2M powf per 960-row launch, measured 84% SM (saturated on
// transcendentals) on the Orin. The table is built ON DEVICE by the same powf
// with the same inputs, so the stored bits — and every downstream angle — are
// identical. Decode's rope_kernel keeps its inline powf (frozen in the graph;
// one row's worth is noise there).
__global__ static void rope_n_kernel(float *v, int half, int hd, const int *d_pos, const float *tab, const float *ff,
                                     int total, int rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total * rows) return;
    int row = idx / total, rem = idx % total;
    int pos = *d_pos + row;
    int head = rem / half, i = rem % half;
    float *vh = v + (size_t)row * (total / half) * hd + (size_t)head * hd;
    float ang = (float)pos * tab[i] / (ff ? ff[i] : 1.0f);
    float c = cosf(ang), s = sinf(ang), a = vh[i], b = vh[i + half];
    vh[i] = a * c - b * s; vh[i + half] = a * s + b * c;
}
__global__ static void rope_tab_kernel(float *tab, float base, int hd, int half) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < half) tab[i] = powf(base, -2.0f * (float)i / (float)hd);
}
// tiny (base, hd) -> device-table cache, built lazily on the prefill path
// (prefill runs uncaptured, so the one-time cudaMalloc + launch are safe).
static const float *rope_tab(float base, int hd) {
    static struct { float base; int hd; float *tab; } slots[4];
    for (int s = 0; s < 4; s++) {
        if (slots[s].tab && slots[s].base == base && slots[s].hd == hd) return slots[s].tab;
        if (!slots[s].tab) {
            CUDA_CHECK(cudaMalloc(&slots[s].tab, (size_t)(hd / 2) * 4));
            rope_tab_kernel<<<gridn(hd / 2), 256>>>(slots[s].tab, base, hd, hd / 2);
            slots[s].base = base; slots[s].hd = hd;
            return slots[s].tab;
        }
    }
    return NULL;  // >4 (base, hd) combos never happens for Gemma; NULL faults loudly
}

// Attention for one query position: one block per query head. pos is device-
// resident and start is derived in-kernel (so the launch params never change across
// tokens -> the forward captures into a static CUDA graph). window=0 means full causal.
// Online softmax: each warp walks its strided slice of timesteps holding a running
// (max, sum) and a per-lane V accumulator in registers, rescaling when the max moves;
// the warps' partials merge once at the end. Shared memory is nwarp*hd — independent
// of how long the context is (the score-array version capped max_seq at ~12k). Scale 1.0.
// The body is shared with the chunk form via (pos, qoff); the decode entry pins
// qoff to 0 so the captured graph keeps the original instruction stream. RING is
// a compile-time flag (one template, separate instruction streams): the ring
// mapping — position start+t lives in cache row (start+t) % seq, T never exceeds
// seq so the modulo is one conditional subtract per timestep — exists ONLY in the
// sliding-window instantiation. The global layers, whose per-timestep loop grows
// with context and notices every added instruction, keep the ring-free body
// (their rows are positions: a full-length buffer never wraps). KT is the cache
// storage type — __half on global layers halves what attention reads per
// timestep, the dominant per-token traffic at a long context; the dot itself
// stays f32 (the conversion rides the load).
#define ATTN_HD_MAX 512   // per-lane V accumulator: ATTN_HD_MAX/32 registers
template <bool RING, typename KT>
__device__ static void d_attn(float *xb, const float *q, const KT *Kc, const KT *Vc,
                              int hd, int kv_dim, int gqa, int pos, size_t qoff, int window, int seq, struct actq aq) {
    int hh = blockIdx.x, kvh = hh / gqa;
    const float *qh = q + qoff + (size_t)hh * hd;
    int start = (window > 0 && pos - window + 1 > 0) ? pos - window + 1 : 0;
    int T = pos - start + 1;
    int s0 = RING ? start % seq : start;
    extern __shared__ float comb[];                       // [nwarp][hd] partial outputs
    __shared__ float wm[32], ws[32];                      // per-warp running max / sum
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = blockDim.x >> 5;
    int nv = hd / 32;                                     // V floats per lane
    float acc[ATTN_HD_MAX / 32];
    #pragma unroll
    for (int j = 0; j < ATTN_HD_MAX / 32; j++) acc[j] = 0.0f;
    float m = -1e30f, s = 0.0f;
    // one WARP per timestep — the lanes split the K dot, so the row is read with
    // coalesced 128-byte loads; the V row follows while the score is still warm.
    for (int t = warp; t < T; t += nwarp) {
        int r = s0 + t;
        if (RING && r >= seq) r -= seq;                   // ring row of position start+t
        const KT *kt = Kc + (size_t)r * kv_dim + (size_t)kvh * hd;
        float sc = 0.0f;
        if (sizeof(KT) == 2) {                            // f16 rows: one 32-bit load = 2 elements
            const __half2 *k2 = (const __half2 *)(const void *)kt;
            for (int i = lane; i < hd / 2; i += 32) {
                float2 kf = __half22float2(k2[i]);
                sc += qh[2 * i] * kf.x + qh[2 * i + 1] * kf.y;
            }
        } else {
            for (int i = lane; i < hd; i += 32) sc += qh[i] * (float)kt[i];
        }
        for (int o = 16; o > 0; o >>= 1) sc += __shfl_down_sync(0xffffffffu, sc, o);
        sc = __shfl_sync(0xffffffffu, sc, 0);
        float mn = fmaxf(m, sc);
        float corr = expf(m - mn), e = expf(sc - mn);     // first step: m=-inf -> corr=0
        s = s * corr + e;
        const KT *vt = Vc + (size_t)r * kv_dim + (size_t)kvh * hd;
        if (sizeof(KT) == 2) {                            // a lane owns element PAIRS here
            const __half2 *v2 = (const __half2 *)(const void *)vt;
            #pragma unroll
            for (int j = 0; j < ATTN_HD_MAX / 64; j++)
                if (j < nv / 2) {
                    float2 vf = __half22float2(v2[lane + j * 32]);
                    acc[2 * j]     = acc[2 * j]     * corr + e * vf.x;
                    acc[2 * j + 1] = acc[2 * j + 1] * corr + e * vf.y;
                }
        } else {
            #pragma unroll
            for (int j = 0; j < ATTN_HD_MAX / 32; j++)
                if (j < nv) acc[j] = acc[j] * corr + e * (float)vt[lane + j * 32];
        }
        m = mn;
    }
    if (lane == 0) { wm[warp] = m; ws[warp] = s; }
    if (sizeof(KT) == 2) {                                // pair layout -> element-indexed comb
        #pragma unroll
        for (int j = 0; j < ATTN_HD_MAX / 64; j++)
            if (j < nv / 2) {
                comb[warp * hd + 2 * (lane + j * 32)]     = acc[2 * j];
                comb[warp * hd + 2 * (lane + j * 32) + 1] = acc[2 * j + 1];
            }
    } else {
        #pragma unroll
        for (int j = 0; j < ATTN_HD_MAX / 32; j++)
            if (j < nv) comb[warp * hd + lane + j * 32] = acc[j];
    }
    __syncthreads();
    // merge the warps: T>=1 keeps warp 0 active, so the global max is finite; an
    // idle warp (T < nwarp) carries m=-inf, s=0 and weighs in as exactly zero.
    float gm = -1e30f;
    for (int w = 0; w < nwarp; w++) gm = fmaxf(gm, wm[w]);
    float gs = 0.0f;
    for (int w = 0; w < nwarp; w++) gs += ws[w] * expf(wm[w] - gm);
    float inv = 1.0f / gs;
    float *outh = xb + qoff + (size_t)hh * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float o = 0.0f;
        for (int w = 0; w < nwarp; w++) o += comb[w * hd + i] * expf(wm[w] - gm);
        outh[i] = o * inv;
    }
    if (aq.xq) {
        __syncthreads();
        for (int g = threadIdx.x; g < hd / 32; g += blockDim.x)
            d_quant_group(outh + g * 32, (int)(qoff / 32) + (hh * hd) / 32 + g, aq);
    }
}
__global__ static void attn_kernel(float *xb, const float *q, const float *Kc, const float *Vc,
                                   int hd, int kv_dim, int gqa, const int *d_pos, int window, struct actq aq) {
    d_attn<false, float>(xb, q, Kc, Vc, hd, kv_dim, gqa, *d_pos, 0, window, 0, aq);
}
__global__ static void attn_h_kernel(float *xb, const float *q, const __half *Kc, const __half *Vc,
                                     int hd, int kv_dim, int gqa, const int *d_pos, int window, struct actq aq) {
    d_attn<false, __half>(xb, q, Kc, Vc, hd, kv_dim, gqa, *d_pos, 0, window, 0, aq);
}
__global__ static void attn_swa_kernel(float *xb, const float *q, const float *Kc, const float *Vc,
                                       int hd, int kv_dim, int gqa, const int *d_pos, int window, int seq, struct actq aq) {
    d_attn<true, float>(xb, q, Kc, Vc, hd, kv_dim, gqa, *d_pos, 0, window, seq, aq);
}
__global__ static void attn_swa_h_kernel(float *xb, const float *q, const __half *Kc, const __half *Vc,
                                         int hd, int kv_dim, int gqa, const int *d_pos, int window, int seq, struct actq aq) {
    d_attn<true, __half>(xb, q, Kc, Vc, hd, kv_dim, gqa, *d_pos, 0, window, seq, aq);
}
// Chunk forms: grid (n_head, chunk) — query blockIdx.y sits at *d_pos + blockIdx.y,
// so a batched chunk is causal by construction: every query reads only positions
// at or before its own, even though the whole chunk's k/v were written before the
// launch (the ring keeps PREFILL_B spare rows so those writes never land on a row
// an earlier query in the chunk still needs).
__global__ static void attn_n_kernel(float *xb, const float *q, const float *Kc, const float *Vc,
                                     int hd, int kv_dim, int gqa, const int *d_pos, int window, struct actq aq) {
    d_attn<false, float>(xb, q, Kc, Vc, hd, kv_dim, gqa, *d_pos + blockIdx.y,
                         (size_t)blockIdx.y * gridDim.x * hd, window, 0, aq);
}
__global__ static void attn_h_n_kernel(float *xb, const float *q, const __half *Kc, const __half *Vc,
                                       int hd, int kv_dim, int gqa, const int *d_pos, int window, struct actq aq) {
    d_attn<false, __half>(xb, q, Kc, Vc, hd, kv_dim, gqa, *d_pos + blockIdx.y,
                          (size_t)blockIdx.y * gridDim.x * hd, window, 0, aq);
}
__global__ static void attn_swa_n_kernel(float *xb, const float *q, const float *Kc, const float *Vc,
                                         int hd, int kv_dim, int gqa, const int *d_pos, int window, int seq, struct actq aq) {
    d_attn<true, float>(xb, q, Kc, Vc, hd, kv_dim, gqa, *d_pos + blockIdx.y,
                        (size_t)blockIdx.y * gridDim.x * hd, window, seq, aq);
}
__global__ static void attn_swa_h_n_kernel(float *xb, const float *q, const __half *Kc, const __half *Vc,
                                           int hd, int kv_dim, int gqa, const int *d_pos, int window, int seq, struct actq aq) {
    d_attn<true, __half>(xb, q, Kc, Vc, hd, kv_dim, gqa, *d_pos + blockIdx.y,
                         (size_t)blockIdx.y * gridDim.x * hd, window, seq, aq);
}

// ==== SPLIT-K (FlashDecoding) decode attention ==============================
// Decode (B=1) attention at a long context is PARALLELISM-bound, not bandwidth-
// bound: d_attn runs one block per head — only n_head blocks on the 8-SM Orin,
// each serially reducing the whole KV (the profile put it ~14x over the KV-
// bandwidth floor). Split the KV reduction across n_split blocks per head: each
// computes a partial (max, sum, V-acc) over its sub-range, then a combine pass
// merges them. n_head x n_split blocks feed the SMs. n_split = clamp(T/SPLIT_KEYS,
// 1, MAXSPLIT), computed IN-KERNEL from the device-resident position, so one
// graph-captured launch (grid is always n_head x MAXSPLIT) covers every context;
// splits beyond n_split early-exit. Same online-softmax math as d_attn,
// deterministic (fixed split order), relaxed class (reassociated). The per-query
// d_attn stays for the B=2 MTP verify and LG_NO_SPLITK fallback.
//
// SPLIT_KEYS is a cap on the KEYS ONE BLOCK WALKS, and that is the whole point:
// it must be small enough that per-block work stays flat as the context grows,
// and the block count absorbs the depth instead. It was 1024 until 2026-07-17,
// which meant n_split==1 for every context under 1024 — the split degenerated to
// one block per head and each block's walk grew LINEARLY with depth. Decode
// drooped -14.3% from shallow to 930-deep on the A5000 (llama.cpp: ~1%, because
// their fattn caps per-block work at nbatch_fa=128 keys and grows the grid).
// At 64: droop -0.6%, and decode +44.7% E2B / +25.5% E4B / +16.7% 12B (A5000),
// +15% / +9.1% (Orin). Sweep: 1024 -> 147.8, 128 -> 203.2, 64 -> 213.9,
// 32 -> 214.5 tok/s (A5000 E2B); the knee is 64 and MAXSPLIT=16 adds nothing
// (64 blocks = 1 CTA/SM = 2 warps/sched is the same balanced-latency floor the
// matmul kernels sit at). The 1024 was tuned on the Orin @4k alone
// (.scratch/flash_decode_test.cu, ~3.3x decode-attn there) where n_head==8
// happens to equal the SM count, so shallow contexts filled the board anyway and
// the droop was invisible. DON'T tune a launch geometry on one device at one
// context depth.
#define MAXSPLIT 8
#define SPLIT_KEYS 64
// blockIdx.z selects the query within a B-row chunk. Decode launches z=1, so qi=0,
// pos=*d_pos, and the (query,head) base collapses to hh*MAXSPLIT — byte-identical
// to the single-query form. The B<=2 MTP verify launches z=B: query qi sits at
// *d_pos+qi and reads its own q row, so verify shares THIS kernel (and reduction
// order) with decode — out[0]/out[1] byte-match plain decode's forwards at
// pos/pos+1 BY CONSTRUCTION (the -mtp==plain invariant no longer rests on a
// numeric near-tie failing to flip between split-K decode and per-query verify).
template <bool RING, typename KT>
__global__ static void split_attn_kernel(float *pacc, float2 *pml, const float *q, const KT *Kc, const KT *Vc,
                                         int hd, int kv_dim, int gqa, const int *d_pos, int window, int seq, int n_head) {
    int hh = blockIdx.x, split = blockIdx.y, qi = blockIdx.z, kvh = hh / gqa;
    int pos = *d_pos + qi, start = (window > 0 && pos - window + 1 > 0) ? pos - window + 1 : 0, T = pos - start + 1;
    int n_split = min(MAXSPLIT, max(1, (T + SPLIT_KEYS - 1) / SPLIT_KEYS));
    size_t hs = ((size_t)qi * n_head + hh) * MAXSPLIT;     // (query,head) base in pacc/pml
    if (split >= n_split) { if (threadIdx.x == 0) pml[hs + split] = make_float2(-1e30f, 0.0f); return; }
    int per = (T + n_split - 1) / n_split, lo = start + split * per, hi = min(pos, lo + per - 1);
    const float *qh = q + (size_t)qi * n_head * hd + (size_t)hh * hd;
    extern __shared__ float comb[];                       // [nwarp][hd]
    __shared__ float wm[32], ws[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = blockDim.x >> 5, nv = hd / 32;
    float acc[ATTN_HD_MAX / 32];
    #pragma unroll
    for (int j = 0; j < ATTN_HD_MAX / 32; j++) acc[j] = 0.0f;
    float m = -1e30f, s = 0.0f;
    for (int t = lo + warp; t <= hi; t += nwarp) {
        int r = RING ? (t >= seq ? t % seq : t) : t;
        const KT *kt = Kc + (size_t)r * kv_dim + (size_t)kvh * hd;
        float sc = 0.0f;
        if (sizeof(KT) == 2) { const __half2 *k2 = (const __half2 *)(const void *)kt;
            for (int i = lane; i < hd / 2; i += 32) { float2 kf = __half22float2(k2[i]); sc += qh[2 * i] * kf.x + qh[2 * i + 1] * kf.y; }
        } else { for (int i = lane; i < hd; i += 32) sc += qh[i] * (float)kt[i]; }
        for (int o = 16; o > 0; o >>= 1) sc += __shfl_down_sync(0xffffffffu, sc, o);
        sc = __shfl_sync(0xffffffffu, sc, 0);
        float mn = fmaxf(m, sc), corr = expf(m - mn), e = expf(sc - mn);
        s = s * corr + e;
        const KT *vt = Vc + (size_t)r * kv_dim + (size_t)kvh * hd;
        if (sizeof(KT) == 2) { const __half2 *v2 = (const __half2 *)(const void *)vt;
            #pragma unroll
            for (int j = 0; j < ATTN_HD_MAX / 64; j++) if (j < nv / 2) { float2 vf = __half22float2(v2[lane + j * 32]);
                acc[2 * j] = acc[2 * j] * corr + e * vf.x; acc[2 * j + 1] = acc[2 * j + 1] * corr + e * vf.y; }
        } else {
            #pragma unroll
            for (int j = 0; j < ATTN_HD_MAX / 32; j++) if (j < nv) acc[j] = acc[j] * corr + e * (float)vt[lane + j * 32];
        }
        m = mn;
    }
    if (lane == 0) { wm[warp] = m; ws[warp] = s; }
    if (sizeof(KT) == 2) {
        #pragma unroll
        for (int j = 0; j < ATTN_HD_MAX / 64; j++) if (j < nv / 2) {
            comb[warp * hd + 2 * (lane + j * 32)] = acc[2 * j]; comb[warp * hd + 2 * (lane + j * 32) + 1] = acc[2 * j + 1]; }
    } else {
        #pragma unroll
        for (int j = 0; j < ATTN_HD_MAX / 32; j++) if (j < nv) comb[warp * hd + lane + j * 32] = acc[j];
    }
    __syncthreads();
    float gm = -1e30f; for (int w = 0; w < nwarp; w++) gm = fmaxf(gm, wm[w]);
    if (threadIdx.x == 0) { float gs = 0.0f; for (int w = 0; w < nwarp; w++) gs += ws[w] * expf(wm[w] - gm);
        pml[hs + split] = make_float2(gm, gs); }
    float *po = pacc + (hs + split) * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float o = 0.0f; for (int w = 0; w < nwarp; w++) o += comb[w * hd + i] * expf(wm[w] - gm);
        po[i] = o;
    }
}
// merge the n_split partials per head -> attention output (+ actq epilogue).
__global__ static void combine_attn_kernel(float *xb, const float *pacc, const float2 *pml,
                                           int hd, const int *d_pos, int window, struct actq aq, int n_head) {
    int hh = blockIdx.x, qi = blockIdx.z;
    int pos = *d_pos + qi, start = (window > 0 && pos - window + 1 > 0) ? pos - window + 1 : 0, T = pos - start + 1;
    int n_split = min(MAXSPLIT, max(1, (T + SPLIT_KEYS - 1) / SPLIT_KEYS));
    size_t hs = ((size_t)qi * n_head + hh) * MAXSPLIT, qoff = (size_t)qi * n_head * hd;
    __shared__ float M, L;
    if (threadIdx.x == 0) { float mm = -1e30f, ll = 0.0f;
        for (int sp = 0; sp < n_split; sp++) { float2 ml = pml[hs + sp]; if (ml.y > 0) mm = fmaxf(mm, ml.x); }
        for (int sp = 0; sp < n_split; sp++) { float2 ml = pml[hs + sp]; if (ml.y > 0) ll += ml.y * expf(ml.x - mm); }
        M = mm; L = ll; }
    __syncthreads();
    float Mv = M, inv = 1.0f / L;
    float *outh = xb + qoff + (size_t)hh * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float o = 0.0f;
        for (int sp = 0; sp < n_split; sp++) { float2 ml = pml[hs + sp];
            if (ml.y > 0) o += pacc[(hs + sp) * hd + i] * expf(ml.x - Mv); }
        outh[i] = o * inv;
    }
    if (aq.xq) { __syncthreads();
        for (int g = threadIdx.x; g < hd / 32; g += blockDim.x) d_quant_group(outh + g * 32, (int)(qoff / 32) + (hh * hd) / 32 + g, aq); }
}
// split-K scratch (partials), grown once during warmup (no mid-capture malloc).
// Sized in (query,head) slots: decode needs n_head (B=1), the B=2 verify 2*n_head.
static float *g_pacc = NULL; static float2 *g_pml = NULL; static int g_split_cap = 0;
static void ensure_split(int slots) {
    if (slots <= g_split_cap) return;
    cudaFree(g_pacc); cudaFree(g_pml);
    CUDA_CHECK(cudaMalloc(&g_pacc, (size_t)slots * MAXSPLIT * ATTN_HD_MAX * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_pml, (size_t)slots * MAXSPLIT * sizeof(float2)));
    g_split_cap = slots;
}

// Write one row (length n) into the kv cache at the device-resident position. Replaces
// a pos-offset cudaMemcpy so the node is static (constant args) and graph-capturable.
__global__ static void kv_write_kernel(float *dst, const float *src, const int *d_pos, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[(size_t)(*d_pos) * n + i] = src[i];
}
// The chunk form: `rows` consecutive cache rows starting at *d_pos.
__global__ static void kv_write_n_kernel(float *dst, const float *src, const int *d_pos, int n, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * rows) dst[(size_t)(*d_pos + i / n) * n + i % n] = src[i];
}
// Ring forms, for sliding-window layers whose cache is seq rows: position p
// lands on row p % seq. Separate kernels so non-ringed layers keep the
// modulo-free originals.
__global__ static void kv_write_ring_kernel(float *dst, const float *src, const int *d_pos, int n, int seq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[(size_t)(*d_pos % seq) * n + i] = src[i];
}
__global__ static void kv_write_ring_n_kernel(float *dst, const float *src, const int *d_pos, int n, int seq, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * rows) dst[(size_t)((*d_pos + i / n) % seq) * n + i % n] = src[i];
}
// f16 forms: the row rounds through half once here — the numerics-changing
// step (round-to-nearest, matching the CPU backend's f16_of bit for bit).
// Originally global layers only; since the f16-SWA-rings step the ring forms
// below cover the sliding-window layers too (same rounding, plus the modulo).
__global__ static void kv_write_h_kernel(__half *dst, const float *src, const int *d_pos, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[(size_t)(*d_pos) * n + i] = __float2half_rn(src[i]);
}
__global__ static void kv_write_h_n_kernel(__half *dst, const float *src, const int *d_pos, int n, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * rows) dst[(size_t)(*d_pos + i / n) * n + i % n] = __float2half_rn(src[i]);
}
__global__ static void kv_write_ring_h_kernel(__half *dst, const float *src, const int *d_pos, int n, int seq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[(size_t)(*d_pos % seq) * n + i] = __float2half_rn(src[i]);
}
__global__ static void kv_write_ring_h_n_kernel(__half *dst, const float *src, const int *d_pos, int n, int seq, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * rows) dst[(size_t)((*d_pos + i / n) % seq) * n + i % n] = __float2half_rn(src[i]);
}

__device__ static float d_gelu(float x) {
    const float k = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}
// Elementwise epilogue: each block covers blockDim.x contiguous elements, so
// after a sync the first few threads can quantize the block's own groups.
__device__ static void d_quant_block(const float *a, int n, struct actq aq) {
    if (!aq.xq) return;
    __syncthreads();
    int base = blockIdx.x * blockDim.x;
    int left = n - base, ng = (left < (int)blockDim.x ? left : (int)blockDim.x) / 32;
    for (int t = threadIdx.x; t < ng; t += blockDim.x)
        d_quant_group(a + base + t * 32, base / 32 + t, aq);
}
__global__ static void geglu_kernel(float *g, const float *u, int n, struct actq aq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = d_gelu(g[i]) * u[i];
    d_quant_block(g, n, aq);
}
// The chunk form: g is [rows][n] contiguous; u's rows may sit ustride apart
// (the PLE inputs are one [n_layer*ple] record per token, so a layer's slice
// strides by the record). Its epilogue quantizes warp-cooperatively (all 8
// warps busy, coalesced) — the serial d_quant_block left 31/32 threads idle
// on the widest activation in the model (rows x n_ff); byte-identical, and
// this kernel is prefill-only so decode's graph never sees the change.
__global__ static void geglu_n_kernel(float *g, const float *u, int n, int rows, int ustride, struct actq aq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * rows) g[i] = d_gelu(g[i]) * u[(size_t)(i / n) * ustride + i % n];
    if (!aq.xq) return;
    __syncthreads();
    int base = blockIdx.x * blockDim.x;
    int left = n * rows - base, ng = (left < (int)blockDim.x ? left : (int)blockDim.x) / 32;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int t = warp; t < ng; t += blockDim.x >> 5)
        d_quant_group_warp(g + base + t * 32, base / 32 + t, aq, lane);
}
__global__ static void scale_const_kernel(float *a, float s, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) a[i] *= s; }
__global__ static void combine_kernel(float *out, const float *p, const float *t, float c, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = (p[i] + t[i]) * c; }
__global__ static void softcap_kernel(float *l, float sc, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) l[i] = sc * tanhf(l[i] / sc); }

// Greedy pick on the device: one 1024-thread block scans the logits. Ties break
// toward the lower index, matching the CPU's first-max scan exactly.
__global__ static void argmax_kernel(const float *x, int n, int *out) {
    __shared__ float bv[1024]; __shared__ int bi[1024];
    float v = -1e30f; int idx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (x[i] > v) { v = x[i]; idx = i; }
    bv[threadIdx.x] = v; bi[threadIdx.x] = idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && (bv[threadIdx.x + s] > bv[threadIdx.x] ||
            (bv[threadIdx.x + s] == bv[threadIdx.x] && bi[threadIdx.x + s] < bi[threadIdx.x]))) {
            bv[threadIdx.x] = bv[threadIdx.x + s]; bi[threadIdx.x] = bi[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = bi[0];
}

// ====================  device state: weights + scratch  =====================

static const struct gguf_context *g_ctx = NULL;
static unsigned char *d_blob = NULL;
static unsigned char **g_dw = NULL;       // per-tensor uploads when the blob never becomes device-resident

// Does this backend repack `type` into its own device copy (so the blob's
// bytes for it are dead after upload)? Defined by the backend .cu below.
static int weights_repacked(uint32_t type);

static size_t tensor_bytes(const struct gguf_tensor *t) {
    int64_t n = 1;
    for (uint32_t d = 0; d < t->n_dims; d++) n *= (int64_t)t->dims[d];
    return (size_t)(n / ggml_blck_size((enum ggml_type)t->type)) * ggml_type_size((enum ggml_type)t->type);
}

static void ensure_weights(struct model *m) {
    ensure_split(LG_MTP_N * m->cfg.n_head);   // split-K scratch: decode B=1 + the B=LG_MTP_N verify; alloc here (pre-capture, no warmup after a chunked prefill)
    if (d_blob || g_dw) return;
    g_ctx = m->ctx;
    // On an integrated GPU (Jetson) host and device share the same DRAM, so
    // copying the blob would hold the weights TWICE in the same physical
    // memory — a 12B that fits fine OOMs at this very malloc. Pin and map the
    // host blob for the GPU instead: zero extra bytes, and decode streams the
    // same LPDDR either way. Discrete GPUs keep the copy (reading weights
    // over PCIe every token would be the opposite of the point).
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    // LG_NO_ZEROCOPY: force the cudaMalloc device copy even on integrated GPUs, to
    // measure the Tegra zero-copy-UNCACHED penalty (mapped host reads bypass L2; a
    // cudaMalloc'd copy is L2-cached). Doubles weight footprint -> only for models
    // that fit twice (E4B 4.6GB ok on 15GB; 12B OOMs).
    static int nozc = -1; if (nozc < 0) nozc = getenv("LG_NO_ZEROCOPY") != NULL;
    if (prop.integrated && !nozc) {
        // When (almost) every weight byte is a type this backend repacks into
        // its own device copy anyway (the QAT E2B is ALL q4_0), pinning the
        // blob holds the quantized payload twice — 2.3GB of dead pinned
        // memory, the difference between fitting and not fitting an 8GB
        // Nano. Upload the residual tensors one by one instead and leave the
        // blob a plain host mapping (embedding dequant and the repack
        // sources still read it). rweight actively hands each repacked
        // tensor's pages back as it copies them (gguf_data_dontneed) —
        // waiting for the OS to evict under pressure was measured NOT to
        // work on Jetson: nvmap fails the allocation instead of reclaiming,
        // and the 12B QAT (6.2 GB of copies vs 6.2 GB of just-read cache)
        // OOM'd at the load boundary.
        size_t inplace = 0;
        for (uint64_t i = 0; i < m->ctx->header.num_tensors; i++) {
            const struct gguf_tensor *t = &m->ctx->tensors[i];
            if (!weights_repacked(t->type)) inplace += tensor_bytes(t);
        }
        if (inplace * 4 <= m->ctx->data_size) {
            g_dw = (unsigned char **)calloc(m->ctx->header.num_tensors, sizeof *g_dw);
            if (!g_dw) { fprintf(stderr, "ensure_weights: out of memory\n"); exit(1); }
            for (uint64_t i = 0; i < m->ctx->header.num_tensors; i++) {
                const struct gguf_tensor *t = &m->ctx->tensors[i];
                if (weights_repacked(t->type)) continue;      // rweight makes its own copy
                size_t bytes = tensor_bytes(t);
                unsigned char *dev;
                CUDA_CHECK(cudaMalloc(&dev, bytes));
                CUDA_CHECK(cudaMemcpy(dev, t->data, bytes, cudaMemcpyHostToDevice));
                g_dw[i] = dev;
            }
            return;
        }
        if (cudaHostRegister(m->ctx->data, m->ctx->data_size, cudaHostRegisterMapped) == cudaSuccess &&
            cudaHostGetDevicePointer((void **)&d_blob, m->ctx->data, 0) == cudaSuccess && d_blob) {
            return;
        }
        cudaGetLastError();                   // clear the failed-register error; fall back
        // Falling through on an integrated GPU means holding the blob TWICE —
        // say so, because for the big models the copy below is what OOMs.
        fprintf(stderr, "weights: zero-copy pin failed, copying the %zu-byte blob (2x memory)\n",
                m->ctx->data_size);
    }
    CUDA_CHECK(cudaMalloc(&d_blob, m->ctx->data_size));
    CUDA_CHECK(cudaMemcpy(d_blob, m->ctx->data, m->ctx->data_size, cudaMemcpyHostToDevice));
}
static const unsigned char *dev_weight(const struct gguf_tensor *t) {
    if (g_dw) return g_dw[(size_t)(t - g_ctx->tensors)];      // NULL only for repacked types, which never ask
    return d_blob + ((const unsigned char *)t->data - (const unsigned char *)g_ctx->data);
}
// Device pointer to an f32 norm/scale weight, or NULL if the tensor is absent.
static const float *dW(struct model *m, const char *name) {
    const struct gguf_tensor *t = gguf_find_tensor(m->ctx, name);
    return t ? (const float *)dev_weight(t) : NULL;
}
static const float *dW_layer(struct model *m, int L, const char *suffix) {
    char nm[96]; snprintf(nm, sizeof nm, "blk.%d.%s", L, suffix);
    return dW(m, nm);
}
static const struct gguf_tensor *wq(struct model *m, const char *name) { return gguf_find_tensor(m->ctx, name); }
static const struct gguf_tensor *wq_layer(struct model *m, int L, const char *suffix) {
    char nm[96]; snprintf(nm, sizeof nm, "blk.%d.%s", L, suffix);
    return gguf_find_tensor(m->ctx, nm);
}

// Which layer's KV layer L uses (KV sharing): own for [0,n_kv_start), else reuse
// the last same-type KV layer (local -> n_kv_start-2, global -> n_kv_start-1).
static int kv_src_dev(const struct model *m, int L) {
    const struct config *c = &m->cfg;
    if (L < c->n_kv_start) return L;
    return c->n_kv_start - (m->is_local[L] ? 2 : 1);
}

// Batched prefill processes the prompt in chunks of this many tokens: each
// weight matrix then crosses the memory bus once per CHUNK instead of once per
// token. The activation buffers below are allocated PREFILL_B rows wide; decode
// and the chunk remainder simply use row 0. Swept 8/16/32 on a 2.4k-token
// prompt (E2B): 327/369/373 tok/s — 16 takes ~99% of 32's rate at half the
// matmul kernel's per-lane accumulators and half the scratch.
// EXPERIMENT (Step 0 of the prefill B-widening; revert to 16 if it loses):
// a 32-wide chunk halves the weight-bus crossings — prefill is bandwidth-bound,
// so chunk width is the dominant lever. The q4_K mma kernel widens to B columns
// in one launch (acc[2][B/8][4]); q6_K stays 16-wide and the dispatch loops it
// B/16 times. B must be a multiple of 16. (The 8/16/32 sweep noted above
// predates the mma kernel and the Orin's zero-copy weights; being re-measured.)
#define PREFILL_B 128
// A whole image span attends bidirectionally, so it must prefill as ONE chunk —
// which can exceed PREFILL_B (up to Gemma's vision ceiling, the 1120-patch budget
// ladder top). PREFILL_MAX_B is the absolute chunk-width cap (stack arrays, kernel
// reach); g_prefill_max_b is the runtime buffer width — PREFILL_B by default, raised
// to the ceiling when a media projector is loaded (model_prefill_reserve), so the
// activation buffers are sized ONCE at scratch init and the decode graph captures
// the final pointers (never realloc'd -> decode untouched). If a big image OOMs,
// that is the deployment's signal to throttle mmcat's -n, not the engine's to cap.
#define PREFILL_MAX_B 1120
static int g_prefill_max_b = PREFILL_B;
extern "C" void model_prefill_reserve(void) {
    static int done = 0; if (done) return; done = 1;
    int w = PREFILL_MAX_B;
    const char *e = getenv("LG_PREFILL_MAX_B");
    if (e) { w = atoi(e); if (w < PREFILL_B) w = PREFILL_B; if (w > PREFILL_MAX_B) w = PREFILL_MAX_B; }
    g_prefill_max_b = w;
}

// LG_WIDE_CHUNK=<N>: prefill TEXT in up-to-N-token chunks (a multiple of 64)
// instead of PREFILL_B=128. Two different wins, one per device class. Integrated
// (Orin): weights are zero-copy host reads, so each weight streams from DRAM once
// per chunk — wider chunk = fewer passes. Discrete (A5000): weights are VRAM-
// resident and the cost is SM fill — a 128-col chunk launches ~30-CTA grids on a
// 64-SM card at 1 CTA/SM, so most of the card idles; the fat kernels tile a wide
// chunk's columns across gridDim.y and fill the machine (measured 2026-07-01:
// 12B warm serve 533 -> ~1200 tok/s). Defaults when unset: 768 on integrated
// (Orin sweep), 1024 on discrete (A5000 sweep; must be a multiple of 64 for the
// single-launch path — PREFILL_MAX_B itself is not). LG_WIDE_CHUNK=128 = legacy.
static int g_wide_chunk = 0;
static void wide_chunk_init(void) {
    static int done = 0; if (done) return; done = 1;
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    const char *e = getenv("LG_WIDE_CHUNK");
    int w = e ? (atoi(e) / 64) * 64 : (p.integrated ? 768 : 1024);
    if (w < PREFILL_B) w = PREFILL_B;
    if (w > PREFILL_MAX_B) w = PREFILL_MAX_B;
    g_wide_chunk = w;
    if (w > g_prefill_max_b) g_prefill_max_b = w;  // size float buffers + KV ring for the wide chunk
}

// Adaptive prefill chunk width. Full chunks use PREFILL_B (128) so long prefills
// (system prompt, media) keep the wide-tile win; the short TAIL of a turn rounds
// up to a multiple of 32 instead of padding to 128 — a 16-token serve turn pays
// a 32-wide chunk, not a 128-wide one (the fat q4_K kernel is templated on COLS,
// and flash / the q6+dp4a sub-tile loops already take the width at runtime).
// matmul_q_n reads this to pick the COLS instantiation; set per chunk by
// forward_chunk*. Buffers stay allocated for the 128 max (see the pre-size in
// model_prefill — g_xq must never realloc after the decode graph is captured).
static int g_pf_cols = PREFILL_B;
// 1 while a B=3 MTP verify chunk is being issued: forces decode's byte-identical
// split-K attention instead of the prefill flash / K-sharing kernels that a B>2
// chunk would otherwise pick (those relax float order and would diverge from plain
// greedy). The B<=2 verify already dodges them via the B<=2 split-K gate.
static int g_chunk_verify = 0;
// Media-span attention context for the current chunk (set by forward_chunk_mixed,
// read where chunk_layers launches flash). g_pf_seg = kv->seg when this chunk holds
// a whole image span (bidirectional within it), NULL otherwise (causal — text,
// decode, verify all leave it NULL). g_pf_bidir_hi = that span's last abs position.
static const int *g_pf_seg = NULL;
static int g_pf_bidir_hi = 0;
// Per-position media-span ids (device), indexed by absolute position; grows to the
// cache capacity. Only the prefill flash path reads it (decode/verify never do), so
// it carries no captured-graph dependency and may realloc freely.
static int *g_seg_dev = NULL; static int g_seg_cap = 0;

// Resident device activation scratch (allocated once, reused across tokens).
static float *dx, *dh, *dq, *dkb, *dvb, *dxb, *dout, *dg1, *dg2, *dpg, *dlogits;
static float *g_hidden;                 // post-output-norm hidden of the last
                                        // head-bearing forward (the MTP draft
                                        // head's h_prev; 15 KB, copied per token)
static float *d_logits_spec;            // MTP verify: logits at the LG_MTP_N rows
static int *d_best_spec;                // MTP verify: argmax of each row
static float *d_ipl, *d_tok, *d_proj;  // per-layer-input (PLE) buffers
static int *d_pos;                      // device-resident token position (for static graph)
static int *d_best;                     // device-side argmax result
static int scratch_ok = 0;

// Fork/join: a graph is a dependency DAG, not a tape. Independent matmuls that
// read the same activation (k+v beside q; up beside gate) are recorded on a side
// stream between two events, so they run CONCURRENTLY in the replayed graph —
// the small latency-bound matmuls hide under the bigger one for free. matmul_q
// launches on g_launch; the forward points it at the side stream inside a fork.
static cudaStream_t g_launch = cudaStreamPerThread, g_side;
static cudaEvent_t g_fork, g_join;
static void side_begin(void) {                          // side stream branches off here
    CUDA_CHECK(cudaEventRecord(g_fork, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamWaitEvent(g_side, g_fork, 0));
    g_launch = g_side;
}
static void side_end(void) {                            // side work recorded; main runs on
    CUDA_CHECK(cudaEventRecord(g_join, g_side));
    g_launch = cudaStreamPerThread;
}
static void side_sync(void) {                           // main needs the side results now
    CUDA_CHECK(cudaStreamWaitEvent(cudaStreamPerThread, g_join, 0));
}

static void ensure_scratch(struct model *m) {
    if (scratch_ok) return;
    wide_chunk_init();            // g_prefill_max_b must be final before B sizes anything
    const struct config *c = &m->cfg;
    int maxhd = 0, maxkv = 0;
    for (int L = 0; L < c->n_layer; L++) {
        if (m->head_dim[L] > maxhd) maxhd = m->head_dim[L];
        int kvd = m->n_head_kv[L] * m->head_dim[L];
        if (kvd > maxkv) maxkv = kvd;
    }
    size_t B = g_prefill_max_b;   // every activation buffer holds a whole prefill chunk (widest = an image span)
    int q_max = c->n_head * maxhd, ne = c->n_embd, nff = c->n_ff, ple = c->n_embd_per_layer;
    CUDA_CHECK(cudaMalloc(&dx,  B * ne   * 4)); CUDA_CHECK(cudaMalloc(&dh,  B * ne   * 4));
    CUDA_CHECK(cudaMalloc(&dout,B * ne   * 4)); CUDA_CHECK(cudaMalloc(&dq,  B * q_max * 4));
    CUDA_CHECK(cudaMalloc(&dxb, B * q_max * 4)); CUDA_CHECK(cudaMalloc(&dkb, B * maxkv * 4));
    CUDA_CHECK(cudaMalloc(&dvb, B * maxkv * 4)); CUDA_CHECK(cudaMalloc(&dg1, B * nff * 4));
    CUDA_CHECK(cudaMalloc(&dg2, B * nff * 4));   CUDA_CHECK(cudaMalloc(&dlogits, (size_t)c->n_vocab * 4));
    CUDA_CHECK(cudaMalloc(&g_hidden, (size_t)ne * 4));
    CUDA_CHECK(cudaMalloc(&d_logits_spec, (size_t)LG_MTP_N * c->n_vocab * 4));   // MTP verify rows
    CUDA_CHECK(cudaMalloc(&d_best_spec, LG_MTP_N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pos, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_best, sizeof(int)));
    if (ple > 0) {
        size_t total = (size_t)ple * c->n_layer;
        CUDA_CHECK(cudaMalloc(&dpg,   B * ple * 4));
        CUDA_CHECK(cudaMalloc(&d_ipl, B * total * 4));
        CUDA_CHECK(cudaMalloc(&d_tok, B * total * 4));
        CUDA_CHECK(cudaMalloc(&d_proj,B * total * 4));
    }
    CUDA_CHECK(cudaStreamCreateWithFlags(&g_side, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&g_fork, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&g_join, cudaEventDisableTiming));
    scratch_ok = 1;
}

// The backend seam — the including .cu defines these four.

// matmul_q: d_out[m] = W . d_x with W the quantized tensor t (all device ptrs).
//   The int8 backend reads d_x's quantization from its buffers, already filled
//   by a producer kernel's epilogue (actq_for) or by act_quantize.
static void matmul_q(float *d_out, const struct gguf_tensor *t, const float *d_x, int k, int m);
// matmul_q_n: the PREFILL_B-column form — d_out[j*m..] = W . d_x[j*k..] for the
//   whole chunk in one launch, so W streams from memory once per chunk.
static void matmul_q_n(float *d_out, const struct gguf_tensor *t, const float *d_x, int k, int m);
// matmul_q_spec: the MTP verify form — d_out[m] = W . d_x for each of the LG_MTP_N 
//  rows that the verify reads, with the same W and d_x as the regular matmul_q. 
static void matmul_q_spec(float *d_out, const struct gguf_tensor *t, const float *d_x, int k, int m);
// matmul_coverage_print: print the per-kernel coverage counts for the matmul_q variants, 
//  to verify that the expected kernels are running for each layer and chunk type.
static void matmul_coverage_print(void);
// actq_for(k): the backend's epilogue buffers for a k-length activation that is
//   about to feed matmul_q (the f32 backend returns AQ0 — no quantization).
static struct actq actq_for(int k);
// act_quantize: quantize d_x explicitly — for the one activation no kernel
//   produces (the host-uploaded embedding); no-op for f32.
static void act_quantize(const float *d_x, int k);
// act_quantize_n: the chunk-wide form (warp-per-group, coalesced). Same output
//   byte-for-byte; separate so decode's captured graph keeps its own kernel.
static void act_quantize_n(const float *d_x, int k);

// ====================  kv cache (device buffers)  ===========================

extern "C" int kvcache_init(struct kvcache *kv, const struct model *m, int max_seq) {
    wide_chunk_init();          // ring spare below must cover the widest chunk this
                                // run can launch; resolve the chunk policy FIRST
                                // (callers with media also call model_prefill_reserve
                                // before this, for the same reason)
    const struct config *c = &m->cfg;
    kv->n_layer = c->n_layer; kv->max_seq = max_seq;
    kv->px_k = kv->px_v = NULL;
    kv->kv_dim = (int *)calloc((size_t)c->n_layer, sizeof(int));
    kv->seq = (int *)calloc((size_t)c->n_layer, sizeof(int));
    kv->f16 = (int *)calloc((size_t)c->n_layer, sizeof(int));
    kv->k = (void **)calloc((size_t)c->n_layer, sizeof(void *));  // host array of device ptrs
    kv->v = (void **)calloc((size_t)c->n_layer, sizeof(void *));
    if (!kv->kv_dim || !kv->seq || !kv->f16 || !kv->k || !kv->v) return -1;
    for (int L = 0; L < c->n_layer; L++) {
        kv->kv_dim[L] = m->n_head_kv[L] * m->head_dim[L];
        if (L >= c->n_kv_start) continue;     // reuses kv_src's buffers: k/v stay NULL
        // A sliding-window layer attends to the last `sliding_window` positions
        // only, so a ring of about that many rows holds everything its queries
        // can reach (row for position p is p % seq). PREFILL_B extra rows keep
        // the batched chunk safe: it writes all B positions BEFORE its queries
        // run, and with a window-exact ring the write for the chunk's last
        // token would land on a row its first query still needs.
        int seq = max_seq;
        // The ring must hold a whole chunk's writes BEYOND the window: a chunk
        // writes all B rows before its queries run, and rows wrap at seq — with
        // seq = window + spare and B <= spare, the highest row a chunk clobbers
        // belongs to a position already outside its first query's window. The
        // spare is g_prefill_max_b, the widest chunk ANY path can launch (text
        // wide chunks and media spans alike); a smaller spare silently corrupts
        // SWA attention for chunks starting past seq (caught 2026-07-01 — the
        // old g_wide_chunk-or-128 spare was also resolved before wide_chunk_init
        // had run, so it was 128 even with LG_WIDE_CHUNK set).
        const int chunk_spare = g_prefill_max_b;
        if (m->is_local[L] && c->sliding_window > 0 && c->sliding_window + chunk_spare < max_seq)
            seq = c->sliding_window + chunk_spare;
        kv->seq[L] = seq;
        // f16 rows everywhere since the f16-SWA-rings step (2026-07): the SWA
        // rings were the last f32 K/V, and they fed the prefill flash 2x the
        // bytes plus an f32->f16 pack per fragment — llama's whole-prefill
        // attention measured 8x ours on the Orin with f16 KV as one of the
        // structural reasons. Numerics-gated like the original f16 global
        // step (determinism + quality, not byte-identity vs f32).
        // LG_SWA_F32=1 restores f32 rings for A/B.
        static int swa_f32 = -1;
        if (swa_f32 < 0) swa_f32 = getenv("LG_SWA_F32") != NULL;
        kv->f16[L] = swa_f32 ? !m->is_local[L] : 1;
        size_t bytes = (size_t)seq * kv->kv_dim[L] * (kv->f16[L] ? 2 : 4);
        CUDA_CHECK(cudaMalloc(&kv->k[L], bytes)); CUDA_CHECK(cudaMemset(kv->k[L], 0, bytes));
        CUDA_CHECK(cudaMalloc(&kv->v[L], bytes)); CUDA_CHECK(cudaMemset(kv->v[L], 0, bytes));
    }
    return 0;
}
extern "C" void kvcache_free(struct kvcache *kv) {
    if (!kv) return;
    for (int L = 0; L < kv->n_layer; L++) { cudaFree(kv->k[L]); cudaFree(kv->v[L]); }
    if (kv->px_k) for (int L = 0; L < kv->n_layer; L++) { cudaFree(kv->px_k[L]); cudaFree(kv->px_v[L]); }
    free(kv->px_k); free(kv->px_v);
    free(kv->k); free(kv->v); free(kv->kv_dim); free(kv->seq); free(kv->f16);
    kv->k = kv->v = NULL; kv->kv_dim = kv->seq = kv->f16 = NULL;
    kv->px_k = kv->px_v = NULL;
}

// System-prefix save/restore (see model.h): device-to-device copies of the
// first min(n, seq) rows of every owning layer. Restore rides the per-thread
// stream, so it is ordered before the session's first prefill launch.
extern "C" void kvcache_save_prefix(struct kvcache *kv, int n) {
    kv->px_k = (void **)calloc((size_t)kv->n_layer, sizeof(void *));
    kv->px_v = (void **)calloc((size_t)kv->n_layer, sizeof(void *));
    if (!kv->px_k || !kv->px_v) return;
    for (int L = 0; L < kv->n_layer; L++) {
        if (!kv->k[L]) continue;
        int rows = n < kv->seq[L] ? n : kv->seq[L];
        size_t bytes = (size_t)rows * kv->kv_dim[L] * (kv->f16[L] ? 2 : 4);
        CUDA_CHECK(cudaMalloc(&kv->px_k[L], bytes));
        CUDA_CHECK(cudaMalloc(&kv->px_v[L], bytes));
        CUDA_CHECK(cudaMemcpy(kv->px_k[L], kv->k[L], bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(kv->px_v[L], kv->v[L], bytes, cudaMemcpyDeviceToDevice));
    }
}
extern "C" void kvcache_restore_prefix(struct kvcache *kv, int n) {
    if (!kv->px_k) return;
    for (int L = 0; L < kv->n_layer; L++) {
        if (!kv->px_k[L]) continue;
        int rows = n < kv->seq[L] ? n : kv->seq[L];
        size_t bytes = (size_t)rows * kv->kv_dim[L] * (kv->f16[L] ? 2 : 4);
        CUDA_CHECK(cudaMemcpyAsync(kv->k[L], kv->px_k[L], bytes, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        CUDA_CHECK(cudaMemcpyAsync(kv->v[L], kv->px_v[L], bytes, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    }
}

// ====================  forward pass (device-resident)  ======================

// Build the PLE inputs on the device; returns 1 if present. dx holds the scaled embedding.
static int build_per_layer(struct model *m, int token) {
    const struct config *c = &m->cfg;
    const int ple = c->n_embd_per_layer;
    if (ple <= 0) return 0;
    const struct gguf_tensor *pte = gguf_find_tensor(m->ctx, "per_layer_token_embd.weight");
    if (!pte) return 0;
    const int64_t total = (int64_t)ple * c->n_layer;

    float *tok = dequantize_row(pte, token, total);   // host
    if (!tok) return 0;
    float te_scale = sqrtf((float)ple);
    for (int64_t i = 0; i < total; i++) tok[i] *= te_scale;
    CUDA_CHECK(cudaMemcpy(d_tok, tok, (size_t)total * 4, cudaMemcpyHostToDevice));
    free(tok);

    act_quantize(dx, c->n_embd);                      // dx came from the host, no producer kernel
    matmul_q(d_proj, wq(m, "per_layer_model_proj.weight"), dx, c->n_embd, (int)total);
    scale_const_kernel<<<gridn((int)total), 256>>>(d_proj, 1.0f / sqrtf((float)c->n_embd), (int)total);
    rmsnorm_kernel<<<c->n_layer, 256>>>(d_proj, d_proj, dW(m, "per_layer_proj_norm.weight"), ple, c->rms_eps, AQ0);
    combine_kernel<<<gridn((int)total), 256>>>(d_ipl, d_proj, d_tok, 1.0f / sqrtf(2.0f), (int)total);
    return 1;
}

// The chunk form: PLE inputs for PREFILL_B tokens — one [n_layer*ple] record
// per token, laid out back to back in d_ipl. Same math per token; the bf16
// projection (the big matmul here) reads its weights once for the whole chunk.
typedef void (*matmul_fn)(float *, const struct gguf_tensor *, const float *, int, int);

static int build_per_layer_n(struct model *m, const int *tokens, int B, matmul_fn mm) {
    const struct config *c = &m->cfg;
    const int ple = c->n_embd_per_layer;
    if (ple <= 0) return 0;
    const struct gguf_tensor *pte = gguf_find_tensor(m->ctx, "per_layer_token_embd.weight");
    if (!pte) return 0;
    const int64_t total = (int64_t)ple * c->n_layer;

    float *host = (float *)malloc((size_t)B * total * 4);
    if (!host) return 0;
    float te_scale = sqrtf((float)ple);
    for (int j = 0; j < B; j++) {
        float *tok = dequantize_row(pte, tokens[j], total);
        if (!tok) { free(host); return 0; }
        for (int64_t i = 0; i < total; i++) host[(size_t)j * total + i] = tok[i] * te_scale;
        free(tok);
    }
    CUDA_CHECK(cudaMemcpy(d_tok, host, (size_t)B * total * 4, cudaMemcpyHostToDevice));
    free(host);

    act_quantize_n(dx, B * c->n_embd);                // dx came from the host, no producer kernel
    mm(d_proj, wq(m, "per_layer_model_proj.weight"), dx, c->n_embd, (int)total);
    scale_const_kernel<<<gridn(B * (int)total), 256>>>(d_proj, 1.0f / sqrtf((float)c->n_embd), B * (int)total);
    rmsnorm_kernel<<<B * c->n_layer, 256>>>(d_proj, d_proj, dW(m, "per_layer_proj_norm.weight"), ple, c->rms_eps, AQ0);
    combine_kernel<<<gridn(B * (int)total), 256>>>(d_ipl, d_proj, d_tok, 1.0f / sqrtf(2.0f), B * (int)total);
    return 1;
}

// Device-only slice of the forward — the part a CUDA graph will capture: all layers
// plus the final norm and output projection. dx already holds the scaled embedding and
// d_ipl the PLE inputs; every op runs on the (per-thread) default stream with no host
// work or sync, so it is stream-capturable. KV-cache writes use async copies for the
// same reason (a synchronous cudaMemcpy is illegal mid-capture).
static void forward_layers(struct model *m, struct kvcache *kv) {
    const struct config *c = &m->cfg;
    const int n_embd = c->n_embd, n_head = c->n_head;
    const float eps = c->rms_eps;
    const int has_ple = c->n_embd_per_layer > 0 &&
                        gguf_find_tensor(m->ctx, "per_layer_token_embd.weight") != NULL;
    const float *d_rope_freqs = dW(m, "rope_freqs.weight");

    for (int L = 0; L < c->n_layer; L++) {
        const int local = m->is_local[L];
        const int hd = m->head_dim[L], n_head_kv = m->n_head_kv[L];
        const int q_dim = n_head * hd, kv_dim = n_head_kv * hd;
        const float base = local ? c->rope_freq_base_swa : c->rope_freq_base;
        const float *ff = local ? NULL : d_rope_freqs;
        const float *os = dW_layer(m, L, "layer_output_scale.weight");  // applied by the layer's last rmsnorm_add

        // ---- attention ----
        rmsnorm_kernel<<<1, NORM_THREADS(n_embd)>>>(dh, dx, dW_layer(m, L, "attn_norm.weight"), n_embd, eps, actq_for(n_embd));

        int src = kv_src_dev(m, L);
        const int has_kv = L < c->n_kv_start;
        if (has_kv) {                                   // k+v matmuls run beside q's
            side_begin();
            matmul_q(dkb, wq_layer(m, L, "attn_k.weight"), dh, n_embd, kv_dim);   // dh still quantized
            const struct gguf_tensor *wv = wq_layer(m, L, "attn_v.weight");
            if (wv) matmul_q(dvb, wv, dh, n_embd, kv_dim);
            else    CUDA_CHECK(cudaMemcpyAsync(dvb, dkb, (size_t)kv_dim * 4, cudaMemcpyDeviceToDevice, g_side));
            side_end();
        }
        matmul_q(dq, wq_layer(m, L, "attn_q.weight"), dh, n_embd, q_dim);
        rmsnorm_kernel<<<n_head, 256>>>(dq, dq, dW_layer(m, L, "attn_q_norm.weight"), hd, eps, AQ0);
        rope_kernel<<<gridn(n_head * hd / 2), 256>>>(dq, hd / 2, hd, d_pos, base, ff, n_head * hd / 2);
        if (has_kv) {
            side_sync();
            rmsnorm_kernel<<<n_head_kv, 256>>>(dkb, dkb, dW_layer(m, L, "attn_k_norm.weight"), hd, eps, AQ0);
            rmsnorm_kernel<<<n_head_kv, 256>>>(dvb, dvb, NULL, hd, eps, AQ0);  // plain V norm
            rope_kernel<<<gridn(n_head_kv * hd / 2), 256>>>(dkb, hd / 2, hd, d_pos, base, ff, n_head_kv * hd / 2);
            if (kv->f16[L] && kv->seq[L] < kv->max_seq) {  // f16 ring (SWA layer)
                kv_write_ring_h_kernel<<<gridn(kv_dim), 256>>>((__half *)kv->k[L], dkb, d_pos, kv_dim, kv->seq[L]);
                kv_write_ring_h_kernel<<<gridn(kv_dim), 256>>>((__half *)kv->v[L], dvb, d_pos, kv_dim, kv->seq[L]);
            } else if (kv->f16[L]) {                       // f16 full-length: row = position
                kv_write_h_kernel<<<gridn(kv_dim), 256>>>((__half *)kv->k[L], dkb, d_pos, kv_dim);
                kv_write_h_kernel<<<gridn(kv_dim), 256>>>((__half *)kv->v[L], dvb, d_pos, kv_dim);
            } else if (kv->seq[L] < kv->max_seq) {         // f32 ring (LG_SWA_F32)
                kv_write_ring_kernel<<<gridn(kv_dim), 256>>>((float *)kv->k[L], dkb, d_pos, kv_dim, kv->seq[L]);
                kv_write_ring_kernel<<<gridn(kv_dim), 256>>>((float *)kv->v[L], dvb, d_pos, kv_dim, kv->seq[L]);
            } else {                                       // full-length f32: row = position
                kv_write_kernel<<<gridn(kv_dim), 256>>>((float *)kv->k[L], dkb, d_pos, kv_dim);
                kv_write_kernel<<<gridn(kv_dim), 256>>>((float *)kv->v[L], dvb, d_pos, kv_dim);
            }
        }
        const void *Kc = kv->k[src], *Vc = kv->v[src];
        int gqa = n_head / n_head_kv;
        int window = (local && c->sliding_window > 0) ? c->sliding_window : 0;
        // shmem holds each warp's partial output row: (256/32)*hd floats, ctx-independent
        size_t shm = (size_t)(256 / 32) * hd * sizeof(float);
        // Split-K decode attention (parallelism, the high-ctx win); d_attn falls
        // back via LG_NO_SPLITK and stays for the B=2 MTP verify. Static grids
        // (graph-capturable); n_split adapts to context on-device.
        static int no_splitk = -1;
        if (no_splitk < 0) no_splitk = getenv("LG_NO_SPLITK") != NULL;
        if (!no_splitk) {
            ensure_split(n_head);
            struct actq aq = actq_for(q_dim);
            dim3 gs(n_head, MAXSPLIT);                      // z=1: qi=0, single query
            int ring = kv->seq[src] < kv->max_seq;
            if (kv->f16[src] && ring)
                split_attn_kernel<true, __half><<<gs, 256, shm>>>(g_pacc, g_pml, dq, (const __half *)Kc, (const __half *)Vc, hd, kv_dim, gqa, d_pos, window, kv->seq[src], n_head);
            else if (kv->f16[src])
                split_attn_kernel<false, __half><<<gs, 256, shm>>>(g_pacc, g_pml, dq, (const __half *)Kc, (const __half *)Vc, hd, kv_dim, gqa, d_pos, window, 0, n_head);
            else if (ring)
                split_attn_kernel<true, float><<<gs, 256, shm>>>(g_pacc, g_pml, dq, (const float *)Kc, (const float *)Vc, hd, kv_dim, gqa, d_pos, window, kv->seq[src], n_head);
            else
                split_attn_kernel<false, float><<<gs, 256, shm>>>(g_pacc, g_pml, dq, (const float *)Kc, (const float *)Vc, hd, kv_dim, gqa, d_pos, window, 0, n_head);
            combine_attn_kernel<<<n_head, 256>>>(dxb, g_pacc, g_pml, hd, d_pos, window, aq, n_head);
        } else if (kv->f16[src] && kv->seq[src] < kv->max_seq)
            attn_swa_h_kernel<<<n_head, 256, shm>>>(dxb, dq, (const __half *)Kc, (const __half *)Vc, hd, kv_dim, gqa, d_pos, window, kv->seq[src], actq_for(q_dim));
        else if (kv->f16[src])
            attn_h_kernel<<<n_head, 256, shm>>>(dxb, dq, (const __half *)Kc, (const __half *)Vc, hd, kv_dim, gqa, d_pos, window, actq_for(q_dim));
        else if (kv->seq[src] < kv->max_seq)
            attn_swa_kernel<<<n_head, 256, shm>>>(dxb, dq, (const float *)Kc, (const float *)Vc, hd, kv_dim, gqa, d_pos, window, kv->seq[src], actq_for(q_dim));
        else
            attn_kernel<<<n_head, 256, shm>>>(dxb, dq, (const float *)Kc, (const float *)Vc, hd, kv_dim, gqa, d_pos, window, actq_for(q_dim));

        matmul_q(dout, wq_layer(m, L, "attn_output.weight"), dxb, q_dim, n_embd);
        rmsnorm_add_kernel<<<1, NORM_THREADS(n_embd)>>>(dx, dout, dW_layer(m, L, "post_attention_norm.weight"), n_embd, eps, NULL, AQ0);

        // ---- feed-forward (GeGLU) ----
        const int nff = m->ffn_len[L];
        rmsnorm_kernel<<<1, NORM_THREADS(n_embd)>>>(dh, dx, dW_layer(m, L, "ffn_norm.weight"), n_embd, eps, actq_for(n_embd));
        side_begin();
        matmul_q(dg2, wq_layer(m, L, "ffn_up.weight"), dh, n_embd, nff);          // dh still quantized
        side_end();
        matmul_q(dg1, wq_layer(m, L, "ffn_gate.weight"), dh, n_embd, nff);
        side_sync();
        geglu_kernel<<<gridn(nff), 256>>>(dg1, dg2, nff, actq_for(nff));
        matmul_q(dout, wq_layer(m, L, "ffn_down.weight"), dg1, nff, n_embd);
        rmsnorm_add_kernel<<<1, NORM_THREADS(n_embd)>>>(dx, dout, dW_layer(m, L, "post_ffw_norm.weight"), n_embd, eps,
                                       has_ple ? NULL : os, has_ple ? actq_for(n_embd) : AQ0);  // PLE reads dx next

        // ---- per-layer input (PLE) ----
        if (has_ple) {
            const int ple = c->n_embd_per_layer;
            matmul_q(dpg, wq_layer(m, L, "inp_gate.weight"), dx, n_embd, ple);
            geglu_kernel<<<gridn(ple), 256>>>(dpg, d_ipl + (size_t)L * ple, ple, actq_for(ple));
            matmul_q(dout, wq_layer(m, L, "proj.weight"), dpg, ple, n_embd);
            rmsnorm_add_kernel<<<1, NORM_THREADS(n_embd)>>>(dx, dout, dW_layer(m, L, "post_norm.weight"), n_embd, eps, os, AQ0);
        }
    }

}

// Final norm + tied output projection. Split from the layers so prefill can
// replay a graph without it: a prompt token's logits are never read, and the
// n_vocab×n_embd head is the single largest matmul in the model.
static void forward_head(struct model *m) {
    const struct config *c = &m->cfg;
    rmsnorm_kernel<<<1, NORM_THREADS(c->n_embd)>>>(dx, dx, dW(m, "output_norm.weight"), c->n_embd, c->rms_eps, actq_for(c->n_embd));
    // keep the post-norm hidden for the MTP draft head (a static graph node;
    // 15 KB D2D, measured no decode cost — see the journal)
    CUDA_CHECK(cudaMemcpyAsync(g_hidden, dx, (size_t)c->n_embd * 4, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    matmul_q(dlogits, wq(m, "token_embd.weight"), dx, c->n_embd, c->n_vocab);
    if (c->logit_softcap > 0.0f)
        softcap_kernel<<<gridn(c->n_vocab), 256>>>(dlogits, c->logit_softcap, c->n_vocab);
}

static void forward_layers_and_head(struct model *m, struct kvcache *kv) {
    forward_layers(m, kv);
    forward_head(m);
}

static int model_has_ple(struct model *m) {
    return m->cfg.n_embd_per_layer > 0 &&
           gguf_find_tensor(m->ctx, "per_layer_token_embd.weight") != NULL;
}

// Capture the device-only forward into a CUDA graph once, then replay it every token:
// ~1000 per-token kernel launches collapse into one graph launch, erasing the WDDM
// launch latency that leaves the GPU idle ~30% of each token. The graph is fully STATIC
// because every per-token-varying input (token position, hence the kv-write offset, the
// rope angle, and the attention range) is read on-device from d_pos, which model_forward
// updates before each launch — so node parameters never change and one exec graph serves
// all tokens. The first two tokens run un-captured so ensure_act's one-time cudaMalloc
// (illegal mid-capture) is already done before capture. (Requires the per-thread default
// stream — see CMakeLists — since the legacy default stream cannot be captured.)
static cudaGraphExec_t g_graph_exec[2] = { NULL, NULL };   // [1]=layers+head (decode), [0]=layers only (prefill)
static int g_graph_warmups = 0;
static void forward_graph(struct model *m, struct kvcache *kv, int head) {
    if (g_graph_warmups < 2) { g_graph_warmups++; forward_layers_and_head(m, kv); return; }
    // Both variants are captured the first time either is needed, so the cost
    // of recording ~1000 nodes lands in the prompt phase, not the gen timer.
    if (!g_graph_exec[0]) {
        for (int h = 0; h < 2; h++) {
            cudaGraph_t graph;
            CUDA_CHECK(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            if (h) forward_layers_and_head(m, kv); else forward_layers(m, kv);
            CUDA_CHECK(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&g_graph_exec[h], graph, 0));
            CUDA_CHECK(cudaGraphDestroy(graph));
        }
    }
    CUDA_CHECK(cudaGraphLaunch(g_graph_exec[!!head], cudaStreamPerThread));
}

static void forward_token(struct model *m, struct kvcache *kv, int token, int pos, int head) {
    ensure_weights(m);
    ensure_scratch(m);
    const struct config *c = &m->cfg;
    const int n_embd = c->n_embd;

    // embedding lookup (host dequant of one row) -> scale -> upload to device
    float *erow = dequantize_row(wq(m, "token_embd.weight"), token, n_embd);
    float es = sqrtf((float)n_embd);
    for (int i = 0; i < n_embd; i++) erow[i] *= es;
    CUDA_CHECK(cudaMemcpy(dx, erow, (size_t)n_embd * 4, cudaMemcpyHostToDevice));
    free(erow);

    CUDA_CHECK(cudaMemcpy(d_pos, &pos, sizeof(int), cudaMemcpyHostToDevice));  // graph reads pos from here
    build_per_layer(m, token);
    forward_graph(m, kv, head);         // capture once (per variant), then replay each token
}

extern "C" void model_forward(struct model *m, struct kvcache *kv, int token, int pos, float *logits) {
    forward_token(m, kv, token, pos, 1);
    CUDA_CHECK(cudaMemcpy(logits, dlogits, (size_t)m->cfg.n_vocab * 4, cudaMemcpyDeviceToHost));
}

// Host-side logits staging for the model_pick hook (-temp): one n_vocab row for
// decode, LG_MTP_N rows for the verify. Only allocated once a hook is set — the
// greedy default never copies logits off the device.
static float *pick_rows(size_t n) {
    static float *buf = NULL;
    static size_t cap = 0;
    if (n > cap) {
        free(buf);
        buf = (float *)malloc(n * 4);
        if (!buf) { fprintf(stderr, "pick_rows: out of memory\n"); exit(1); }
        cap = n;
    }
    return buf;
}

extern "C" int model_forward_next(struct model *m, struct kvcache *kv, int token, int pos) {
    forward_token(m, kv, token, pos, 1);
    if (model_pick) {
        float *row = pick_rows(m->cfg.n_vocab);
        CUDA_CHECK(cudaMemcpy(row, dlogits, (size_t)m->cfg.n_vocab * 4, cudaMemcpyDeviceToHost));
        return model_pick(row, m->cfg.n_vocab);
    }
    argmax_kernel<<<1, 1024>>>(dlogits, m->cfg.n_vocab, d_best);
    int best;
    CUDA_CHECK(cudaMemcpy(&best, d_best, sizeof(int), cudaMemcpyDeviceToHost));
    return best;
}

#include "prefill-kernel.cuh"                 // chunked prefill: flash, K-share, model_prefill*
#include "mtp-kernel.cuh"                   // MTP verify + device draft (split for size)

#endif // MODEL_CUDA_CUH
