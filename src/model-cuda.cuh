// Shared CUDA backend: everything except the matmul. Included by exactly one
// compute file per binary — model-cuda-f32.cu (f32 dequant dot) or
// model-cuda-i8r.cu (int8 dot) — each of which defines matmul_q. The forward
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
__global__ static void rope_n_kernel(float *v, int half, int hd, const int *d_pos, float base, const float *ff,
                                     int total, int rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total * rows) return;
    int row = idx / total, rem = idx % total;
    int pos = *d_pos + row;
    int head = rem / half, i = rem % half;
    float *vh = v + (size_t)row * (total / half) * hd + (size_t)head * hd;
    float freq = powf(base, -2.0f * (float)i / (float)hd);
    float ang = (float)pos * freq / (ff ? ff[i] : 1.0f);
    float c = cosf(ang), s = sinf(ang), a = vh[i], b = vh[i + half];
    vh[i] = a * c - b * s; vh[i + half] = a * s + b * c;
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
    float m = -INFINITY, s = 0.0f;
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
    float gm = -INFINITY;
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

// ==== SPLIT-K (FlashDecoding) decode attention ==============================
// Decode (B=1) attention at a long context is PARALLELISM-bound, not bandwidth-
// bound: d_attn runs one block per head — only n_head blocks on the 8-SM Orin,
// each serially reducing the whole KV (the profile put it ~14x over the KV-
// bandwidth floor). Split the KV reduction across n_split blocks per head: each
// computes a partial (max, sum, V-acc) over its sub-range, then a combine pass
// merges them. n_head x n_split blocks feed the SMs. n_split is device-adaptive
// (clamp(T/SPLIT_KEYS,1,MAXSPLIT)) so one graph-captured launch covers all
// contexts; splits beyond n_split early-exit. Same online-softmax math as
// d_attn, deterministic (fixed split order), relaxed class (reassociated). The
// per-query d_attn stays for the B=2 MTP verify and LG_NO_SPLITK fallback.
// Measured (.scratch/flash_decode_test.cu) ~3.3x decode-attn on the Orin @4k.
#define MAXSPLIT 8
#define SPLIT_KEYS 1024
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
    if (split >= n_split) { if (threadIdx.x == 0) pml[hs + split] = make_float2(-INFINITY, 0.0f); return; }
    int per = (T + n_split - 1) / n_split, lo = start + split * per, hi = min(pos, lo + per - 1);
    const float *qh = q + (size_t)qi * n_head * hd + (size_t)hh * hd;
    extern __shared__ float comb[];                       // [nwarp][hd]
    __shared__ float wm[32], ws[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = blockDim.x >> 5, nv = hd / 32;
    float acc[ATTN_HD_MAX / 32];
    #pragma unroll
    for (int j = 0; j < ATTN_HD_MAX / 32; j++) acc[j] = 0.0f;
    float m = -INFINITY, s = 0.0f;
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
    float gm = -INFINITY; for (int w = 0; w < nwarp; w++) gm = fmaxf(gm, wm[w]);
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
    if (threadIdx.x == 0) { float mm = -INFINITY, ll = 0.0f;
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

// K/V-SHARING prefill attention. The per-query kernels above launch one block
// per (head, query); each re-reads the whole K/V prefix, so a chunk's B queries
// read it B times — and at a long context K/V exceeds L2, so that redundancy is
// the dominant prefill-attention cost (the profile put it at 28% E4B / 43% 12B).
// Here a block owns a tile of QT queries for one head: each K/V row is staged
// into shared ONCE per block and reused by all QT warps (one warp per query,
// online-softmax state in registers, exactly like d_attn). Global K/V reads
// drop ~QT×; the math per query is unchanged (validated in .scratch/
// attn_kvshare_test.cu vs a double reference). PREFILL ONLY — gated on B>2 so
// the B=2 MTP verify keeps the per-query path and its decode-matching argmax.
// The float combine order differs from the per-query kernel (one warp now sums
// all timesteps of its query, vs warps splitting them), so this is the same
// relaxed class as the online-softmax step; the quantize epilogue calls the
// identical d_quant_group, so only attention's own reassociation differs.
template <int QT, bool RING, typename KT>
__global__ static void attn_kvshare_n_kernel(float *xb, const float *q, const KT *Kc, const KT *Vc,
                                             int hd, int kv_dim, int gqa, const int *d_pos, int window, int seq,
                                             int B, int n_head, struct actq aq) {
    const int hh = blockIdx.x, kvh = hh / gqa;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int qbase = blockIdx.y * QT, b = qbase + warp;  // this warp's query (chunk row)
    const int pos = *d_pos + b;                           // its absolute position (valid if b<B)
    const int start = (window > 0 && pos - window + 1 > 0) ? pos - window + 1 : 0;
    const int nv = hd / 32;
    extern __shared__ float katt[];                       // sK[hd], sV[hd], sO[QT*hd]
    float *sK = katt, *sV = katt + hd, *sO = katt + 2 * hd;
    const float *qh = q + (size_t)b * n_head * hd + (size_t)hh * hd;   // deref only when b<B
    float acc[ATTN_HD_MAX / 32];
    #pragma unroll
    for (int j = 0; j < ATTN_HD_MAX / 32; j++) acc[j] = 0.0f;
    float m = -INFINITY, s = 0.0f;

    // positions this block streams: from the tile's earliest window-start to its
    // latest query position. (For full layers window==0 -> from 0.)
    int lastb = qbase + QT - 1; if (lastb > B - 1) lastb = B - 1;
    int rhi = *d_pos + lastb;
    int rlo = 0;
    if (RING && window > 0) { rlo = *d_pos + qbase - window + 1; if (rlo < 0) rlo = 0; }
    for (int r = rlo; r <= rhi; r++) {
        int rr = RING ? r % seq : r;                      // ring row of absolute position r
        if (sizeof(KT) == 2) {                            // f16: one 32-bit load = 2 elements
            const __half2 *K2 = (const __half2 *)(const void *)(Kc + (size_t)rr * kv_dim + (size_t)kvh * hd);
            const __half2 *V2 = (const __half2 *)(const void *)(Vc + (size_t)rr * kv_dim + (size_t)kvh * hd);
            for (int i = threadIdx.x; i < hd / 2; i += blockDim.x) {
                float2 kf = __half22float2(K2[i]); sK[2 * i] = kf.x; sK[2 * i + 1] = kf.y;
                float2 vf = __half22float2(V2[i]); sV[2 * i] = vf.x; sV[2 * i + 1] = vf.y;
            }
        } else {
            const KT *kt = Kc + (size_t)rr * kv_dim + (size_t)kvh * hd;
            const KT *vt = Vc + (size_t)rr * kv_dim + (size_t)kvh * hd;
            for (int i = threadIdx.x; i < hd; i += blockDim.x) { sK[i] = (float)kt[i]; sV[i] = (float)vt[i]; }
        }
        __syncthreads();
        if (b < B && r >= start && r <= pos) {            // warp-uniform: causal + window mask
            float sc = 0.0f;
            for (int i = lane; i < hd; i += 32) sc += qh[i] * sK[i];
            for (int o = 16; o > 0; o >>= 1) sc += __shfl_down_sync(0xffffffffu, sc, o);
            sc = __shfl_sync(0xffffffffu, sc, 0);
            float mn = fmaxf(m, sc), corr = expf(m - mn), e = expf(sc - mn);
            s = s * corr + e;
            #pragma unroll
            for (int j = 0; j < ATTN_HD_MAX / 32; j++) if (j < nv) acc[j] = acc[j] * corr + e * sV[lane + j * 32];
            m = mn;
        }
        __syncthreads();                                  // before next r overwrites sK/sV
    }

    if (b < B) {
        float inv = 1.0f / s;
        float *outh = xb + (size_t)b * n_head * hd + (size_t)hh * hd;
        #pragma unroll
        for (int j = 0; j < ATTN_HD_MAX / 32; j++)
            if (j < nv) { float v = acc[j] * inv; outh[lane + j * 32] = v; sO[warp * hd + lane + j * 32] = v; }
        __syncwarp();                                     // sO visible across the warp
        if (aq.xq) {
            int gbase = (int)(((size_t)b * n_head * hd + (size_t)hh * hd) / 32);
            for (int g = lane; g < nv; g += 32) d_quant_group(sO + warp * hd + g * 32, gbase + g, aq);
        }
    }
}

// ==== tensor-core FLASH ATTENTION (prefill, B>2) ============================
// The attn_*_n kernels above are online-softmax with SCALAR dots — on the Orin
// prefill profile, attention is ~40% of TTFT and those dots leave the tensor
// cores idle. This is our specialized TC flash attention: per (query,head),
// out = softmax(Q.K^T).V (scale 1.0, no softcap, causal/windowed), Q/K/V already
// normed+roped. One CTA per head, 8 warps. KEY Orin choice: K/V are NOT staged
// to shared (read from the L2-cached cache) so shared stays small (~38KB at
// hd=512) and the 8-SM Orin seats ~4 CTAs — avoiding the 1-CTA/SM occupancy trap
// that sank the tiled MMQ. 8-warp QK^T computes one S[16q x 8k] tile FULL-hd (no
// cross-warp reduce) -> shared; online softmax (warp per 4 query rows); PV is
// hd-SPLIT (warp w owns hd[w*HDW,+HDW)) with V transposed via scalar reads.
// m16n8k16.f16.f16.f32: Q,P and the cache rows feed the mma as f16 (f32 SWA KV
// is converted) — precision is the f16-flash/quality-equivalent class (same as
// the shipped f16-KV step, ~2e-3 vs an f64 ref, validated in .scratch/flash_test
// .cu), NOT bit-identical. Writes f32 xb; the dispatch quantizes via act_quantize
// (the C-tile scatter doesn't give the contiguous 32-groups the epilogue needs).
// PREFILL ONLY, B>2 — the B<=2 MTP verify keeps the per-query path (bit-exact
// argmax). LG_NO_FLASH falls back to the attn_*_n kernels.
template<typename KT> __device__ __forceinline__ uint32_t fa_ld2(const KT *p);
template<> __device__ __forceinline__ uint32_t fa_ld2<__half>(const __half *p){ return *(const uint32_t *)p; }
template<> __device__ __forceinline__ uint32_t fa_ld2<float>(const float *p){
    return (uint32_t)__half_as_ushort(__float2half(p[0])) | ((uint32_t)__half_as_ushort(__float2half(p[1]))<<16); }
template<typename KT> __device__ __forceinline__ __half fa_rd1(const KT *p);
template<> __device__ __forceinline__ __half fa_rd1<__half>(const __half *p){ return *p; }
template<> __device__ __forceinline__ __half fa_rd1<float>(const float *p){ return __float2half(*p); }
__device__ __forceinline__ uint32_t fa_pk(__half a,__half b){ return (uint32_t)__half_as_ushort(a) | ((uint32_t)__half_as_ushort(b)<<16); }
__device__ __forceinline__ void fa_mma(float &c0,float &c1,float &c2,float &c3,
        uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1)); }

// seg/bidir_hi carry Gemma's modality mask: seg[abs_pos] is a media-span id (0 for
// text), and within a span attention is BIDIRECTIONAL (a patch sees its whole frame,
// not just earlier patches). seg==NULL -> the kernel is byte-identical causal (text,
// decode, verify). For a media chunk the key loop extends to bidir_hi (the span's
// last position, which can sit past a query's own position) and same-span keys are
// unmasked regardless of causal/window — that is the frame attending to itself.
template<int HD, bool RING, typename KT>
__global__ static void __launch_bounds__(256)
flash_attn_n_kernel(float *xb, const float *q, const KT *Kc, const KT *Vc,
                    int kv_dim, int gqa, const int *d_pos, int window, int seq, int B, int n_head,
                    const int *seg, int bidir_hi){
    const int FKB = 32, FHDW = HD/8;                       // keys/block; PV hd-slice per warp
    __shared__ __half sQ[32*HD];
    __shared__ float  sS[32*FKB];
    __shared__ __half sP[32*FKB];
    __shared__ float  sm[32], sl[32], sc[32];
    const int head=blockIdx.x, warp=threadIdx.x>>5, lane=threadIdx.x&31, tix=threadIdx.x;
    const int gid=lane>>2, tid=lane&3, kvh=head/gqa, pos0=*d_pos;
    const int qbase=blockIdx.y*32, qn=(B-qbase<32)?(B-qbase):32;   // this CTA's 32-query tile (blockIdx.y); B>32 tiles over y
    float acc[2][FHDW/8][4];
    #pragma unroll
    for(int m=0;m<2;m++)for(int n=0;n<FHDW/8;n++)for(int c=0;c<4;c++)acc[m][n][c]=0.0f;
    for(int Q=tix;Q<32;Q+=256){ sm[Q]=-1e30f; sl[Q]=0.0f; }
    for(int t=tix;t<32*HD;t+=256){ int b=t/HD,i=t%HD;
        sQ[t]=(b<qn)?__float2half(q[(size_t)(qbase+b)*n_head*HD + (size_t)head*HD + i]):(__half)0; }
    __syncthreads();
    int pmax=pos0+qbase+qn-1, smin=(window>0 && pos0+qbase-window+1>0)?pos0+qbase-window+1:0;
    int khi = (seg && bidir_hi > pmax) ? bidir_hi : pmax;   // media chunk: reach past this tile to the span end
    for(int kb0=(smin/FKB)*FKB; kb0<=khi; kb0+=FKB){
        int qt=warp>>2, kt=warp&3;
        float s0=0,s1=0,s2=0,s3=0;
        #pragma unroll
        for(int ks=0;ks<HD;ks+=16){
            uint32_t a0=*(uint32_t*)&sQ[(qt*16+gid)*HD+ks+tid*2],   a1=*(uint32_t*)&sQ[(qt*16+gid+8)*HD+ks+tid*2];
            uint32_t a2=*(uint32_t*)&sQ[(qt*16+gid)*HD+ks+tid*2+8], a3=*(uint32_t*)&sQ[(qt*16+gid+8)*HD+ks+tid*2+8];
            int key=kb0+kt*8+gid, rr=RING?key%seq:key;
            const KT *kp=Kc+(size_t)rr*kv_dim+(size_t)kvh*HD+ks;
            fa_mma(s0,s1,s2,s3,a0,a1,a2,a3,fa_ld2<KT>(kp+tid*2),fa_ld2<KT>(kp+tid*2+8));
        }
        sS[(qt*16+gid)*FKB+kt*8+tid*2]=s0;     sS[(qt*16+gid)*FKB+kt*8+tid*2+1]=s1;
        sS[(qt*16+gid+8)*FKB+kt*8+tid*2]=s2;   sS[(qt*16+gid+8)*FKB+kt*8+tid*2+1]=s3;
        __syncthreads();
        #pragma unroll
        for(int r=0;r<4;r++){
            int b=warp*4+r, pos=pos0+qbase+b, start=(window>0&&pos-window+1>0)?pos-window+1:0;
            int kabs=kb0+lane; bool ok=(lane<FKB)&&(b<qn)&&(kabs>=start)&&(kabs<=pos);
            if(seg && !ok && lane<FKB && b<qn && kabs<=bidir_hi){   // same media span: bidirectional (bypasses causal+window)
                int sq=seg[pos]; ok=(sq!=0)&&(sq==seg[kabs]); }
            float s=ok?sS[b*FKB+lane]:-1e30f, rmax=s;
            for(int o=16;o>0;o>>=1) rmax=fmaxf(rmax,__shfl_xor_sync(~0u,rmax,o));
            float mn=fmaxf(sm[b],rmax), corr=__expf(sm[b]-mn), p=ok?__expf(s-mn):0.0f, rsum=p;
            for(int o=16;o>0;o>>=1) rsum+=__shfl_xor_sync(~0u,rsum,o);
            if(lane<FKB) sP[b*FKB+lane]=__float2half(p);
            if(lane==0){ sl[b]=sl[b]*corr+rsum; sm[b]=mn; sc[b]=corr; }
        }
        __syncthreads();
        #pragma unroll
        for(int m=0;m<2;m++){ float cA=sc[m*16+gid], cB=sc[m*16+gid+8];
            for(int n=0;n<FHDW/8;n++){ acc[m][n][0]*=cA; acc[m][n][1]*=cA; acc[m][n][2]*=cB; acc[m][n][3]*=cB; } }
        // PV: V is query-INDEPENDENT, so read each V fragment ONCE and feed both
        // query-tiles (m) — halves the (uncoalesced, column-strided) V global reads.
        #pragma unroll
        for(int ks2=0;ks2<FKB;ks2+=16){
            uint32_t pa[2][4];
            #pragma unroll
            for(int m=0;m<2;m++){
                pa[m][0]=*(uint32_t*)&sP[(m*16+gid)*FKB+ks2+tid*2];   pa[m][1]=*(uint32_t*)&sP[(m*16+gid+8)*FKB+ks2+tid*2];
                pa[m][2]=*(uint32_t*)&sP[(m*16+gid)*FKB+ks2+tid*2+8]; pa[m][3]=*(uint32_t*)&sP[(m*16+gid+8)*FKB+ks2+tid*2+8];
            }
            int k0=kb0+ks2+tid*2;
            int r0=RING?k0%seq:k0, r1=RING?(k0+1)%seq:k0+1, r8=RING?(k0+8)%seq:k0+8, r9=RING?(k0+9)%seq:k0+9;
            #pragma unroll
            for(int n=0;n<FHDW/8;n++){ int hdn=warp*FHDW+n*8+gid;
                const KT *vb=Vc+(size_t)kvh*HD+hdn;
                uint32_t b0=fa_pk(fa_rd1<KT>(vb+(size_t)r0*kv_dim), fa_rd1<KT>(vb+(size_t)r1*kv_dim));
                uint32_t b1=fa_pk(fa_rd1<KT>(vb+(size_t)r8*kv_dim), fa_rd1<KT>(vb+(size_t)r9*kv_dim));
                #pragma unroll
                for(int m=0;m<2;m++)
                    fa_mma(acc[m][n][0],acc[m][n][1],acc[m][n][2],acc[m][n][3],pa[m][0],pa[m][1],pa[m][2],pa[m][3],b0,b1);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int m=0;m<2;m++) for(int n=0;n<FHDW/8;n++){
        int qA=m*16+gid, qB=m*16+gid+8, hdc=warp*FHDW+n*8+tid*2;
        if(qA<qn){ xb[((size_t)(qbase+qA)*n_head+head)*HD+hdc]=acc[m][n][0]/sl[qA]; xb[((size_t)(qbase+qA)*n_head+head)*HD+hdc+1]=acc[m][n][1]/sl[qA]; }
        if(qB<qn){ xb[((size_t)(qbase+qB)*n_head+head)*HD+hdc]=acc[m][n][2]/sl[qB]; xb[((size_t)(qbase+qB)*n_head+head)*HD+hdc+1]=acc[m][n][3]/sl[qB]; }
    }
}
// hd is a compile-time template (256 SWA / 512 global); pick KT/RING like the
// per-query dispatch: f16 -> global (no ring), f32+seq<max -> SWA ring, else full f32.
template<int HD>
static void launch_flash(float *dxb, const float *dq, const void *Kc, const void *Vc,
                         int kv_dim, int gqa, const int *d_pos, int window, int seq, int B, int n_head,
                         bool f16, bool ring, const int *seg, int bidir_hi) {
    dim3 g(n_head, (B + 31) / 32);                         // y = 32-query tiles (B>32 prefill chunks)
    if (f16)      flash_attn_n_kernel<HD,false,__half><<<g,256>>>(dxb,dq,(const __half*)Kc,(const __half*)Vc,kv_dim,gqa,d_pos,window,seq,B,n_head,seg,bidir_hi);
    else if (ring) flash_attn_n_kernel<HD,true, float ><<<g,256>>>(dxb,dq,(const float*)Kc,(const float*)Vc,kv_dim,gqa,d_pos,window,seq,B,n_head,seg,bidir_hi);
    else          flash_attn_n_kernel<HD,false,float ><<<g,256>>>(dxb,dq,(const float*)Kc,(const float*)Vc,kv_dim,gqa,d_pos,window,seq,B,n_head,seg,bidir_hi);
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
// f16 forms, for global layers: the row rounds through half once here — the
// single numerics-changing step in the pipeline (round-to-nearest, matching
// the CPU backend's f16_of bit for bit). Global layers never ring.
__global__ static void kv_write_h_kernel(__half *dst, const float *src, const int *d_pos, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[(size_t)(*d_pos) * n + i] = __float2half_rn(src[i]);
}
__global__ static void kv_write_h_n_kernel(__half *dst, const float *src, const int *d_pos, int n, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * rows) dst[(size_t)(*d_pos + i / n) * n + i % n] = __float2half_rn(src[i]);
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
// strides by the record).
__global__ static void geglu_n_kernel(float *g, const float *u, int n, int rows, int ustride, struct actq aq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * rows) g[i] = d_gelu(g[i]) * u[(size_t)(i / n) * ustride + i % n];
    d_quant_block(g, n * rows, aq);
}
__global__ static void scale_const_kernel(float *a, float s, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) a[i] *= s; }
__global__ static void combine_kernel(float *out, const float *p, const float *t, float c, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = (p[i] + t[i]) * c; }
__global__ static void softcap_kernel(float *l, float sc, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) l[i] = sc * tanhf(l[i] / sc); }

// Greedy pick on the device: one 1024-thread block scans the logits. Ties break
// toward the lower index, matching the CPU's first-max scan exactly.
__global__ static void argmax_kernel(const float *x, int n, int *out) {
    __shared__ float bv[1024]; __shared__ int bi[1024];
    float v = -INFINITY; int idx = 0;
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

static void ensure_weights(struct model *m) {
    ensure_split(LG_MTP_N * m->cfg.n_head);   // split-K scratch: decode B=1 + the B=LG_MTP_N verify; alloc here (pre-capture, no warmup after a chunked prefill)
    if (d_blob) return;
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
    if (prop.integrated && !nozc &&
        cudaHostRegister(m->ctx->data, m->ctx->data_size, cudaHostRegisterMapped) == cudaSuccess &&
        cudaHostGetDevicePointer((void **)&d_blob, m->ctx->data, 0) == cudaSuccess && d_blob) {
        return;
    }
    cudaGetLastError();                       // clear any failed-register error; fall back
    CUDA_CHECK(cudaMalloc(&d_blob, m->ctx->data_size));
    CUDA_CHECK(cudaMemcpy(d_blob, m->ctx->data, m->ctx->data_size, cudaMemcpyHostToDevice));
}
static const unsigned char *dev_weight(const struct gguf_tensor *t) {
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

// LG_WIDE_CHUNK=<N>: prefill TEXT in N-token chunks (a multiple of 128) instead of
// PREFILL_B=128, so each model weight streams from DRAM once per N tokens, not once
// per 128. The fat q4_K/q6_K kernels then tile the chunk's columns across gridDim.y
// (COLS stays 128/64 per block — no 1-CTA trap) so a row-tile's weights stay L2-hot
// across its column-tiles, matching llama.cpp's 512-ubatch weight reuse. The wide
// chunk path through attention/ring/buffers already exists (media spans up to
// g_prefill_max_b); this just opens it to text and widens the buffer/ring reserve.
// 0 = off (legacy 128-token chunking, byte-identical default).
static int g_wide_chunk = 0;
static void wide_chunk_init(void) {
    static int done = 0; if (done) return; done = 1;
    const char *e = getenv("LG_WIDE_CHUNK");
    if (!e) return;
    int w = (atoi(e) / 128) * 128;                 // a whole number of 128-col tiles
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

// ====================  kv cache (device buffers)  ===========================

extern "C" int kvcache_init(struct kvcache *kv, const struct model *m, int max_seq) {
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
        const int chunk_spare = g_wide_chunk ? g_wide_chunk : PREFILL_B;  // ring must hold a whole chunk's writes
        if (m->is_local[L] && c->sliding_window > 0 && c->sliding_window + chunk_spare < max_seq)
            seq = c->sliding_window + chunk_spare;
        kv->seq[L] = seq;
        kv->f16[L] = !m->is_local[L];         // global rows are f16 (see model.h)
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

    act_quantize(dx, B * c->n_embd);                  // dx came from the host, no producer kernel
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
            if (kv->f16[L]) {                              // global layer: rows round through f16
                kv_write_h_kernel<<<gridn(kv_dim), 256>>>((__half *)kv->k[L], dkb, d_pos, kv_dim);
                kv_write_h_kernel<<<gridn(kv_dim), 256>>>((__half *)kv->v[L], dvb, d_pos, kv_dim);
            } else if (kv->seq[L] < kv->max_seq) {         // ring (sliding-window layer)
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
            if (kv->f16[src])
                split_attn_kernel<false, __half><<<gs, 256, shm>>>(g_pacc, g_pml, dq, (const __half *)Kc, (const __half *)Vc, hd, kv_dim, gqa, d_pos, window, 0, n_head);
            else if (kv->seq[src] < kv->max_seq)
                split_attn_kernel<true, float><<<gs, 256, shm>>>(g_pacc, g_pml, dq, (const float *)Kc, (const float *)Vc, hd, kv_dim, gqa, d_pos, window, kv->seq[src], n_head);
            else
                split_attn_kernel<false, float><<<gs, 256, shm>>>(g_pacc, g_pml, dq, (const float *)Kc, (const float *)Vc, hd, kv_dim, gqa, d_pos, window, 0, n_head);
            combine_attn_kernel<<<n_head, 256>>>(dxb, g_pacc, g_pml, hd, d_pos, window, aq, n_head);
        } else if (kv->f16[src])
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

// One prefill chunk's layer loop: the forward for the PREFILL_B positions
// whose inputs already sit in dx (and whose PLE rows, if the model has them,
// sit in d_ipl), at positions *d_pos..*d_pos+B-1, head skipped. The math is
// exactly B single-token forwards — every kernel above carries a row dimension
// that decode launches at 1 — but each weight matrix crosses the memory bus
// ONCE per chunk instead of once per token, and prefill is bandwidth-bound,
// so that factor is most of its cost. Per layer, the whole chunk's k/v are
// written to the cache before its queries run; causality holds because each
// query reads only up to its own position. Runs un-captured (and without the
// fork/join forks): a chunk already amortizes launch latency over its B tokens.
// LG_PREFILL_PROFILE=1: per-stage wall time across a whole prefill, with a
// sync after each stage group (slows the run; for attribution only).
static double g_pf_mm = 0, g_pf_attn = 0, g_pf_elem = 0, g_pf_ple = 0;
static int g_pf_on = -1;
static int pf_on(void) {
    if (g_pf_on < 0) g_pf_on = getenv("LG_PREFILL_PROFILE") != NULL;
    return g_pf_on;
}
static double pf_tick(double *acc) {
    static double last;
    if (!pf_on()) return 0;
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    double now;
    { struct timespec ts; timespec_get(&ts, TIME_UTC); now = (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec; }
    if (acc) *acc += now - last;
    last = now;
    return now;
}

static void chunk_layers(struct model *m, struct kvcache *kv, int has_ple, int B, matmul_fn mm) {
    const struct config *c = &m->cfg;
    const int n_embd = c->n_embd, n_head = c->n_head;
    const float eps = c->rms_eps;
    const float *d_rope_freqs = dW(m, "rope_freqs.weight");

    pf_tick(NULL);
    for (int L = 0; L < c->n_layer; L++) {
        const int local = m->is_local[L];
        const int hd = m->head_dim[L], n_head_kv = m->n_head_kv[L];
        const int q_dim = n_head * hd, kv_dim = n_head_kv * hd;
        const float base = local ? c->rope_freq_base_swa : c->rope_freq_base;
        const float *ff = local ? NULL : d_rope_freqs;
        const float *os = dW_layer(m, L, "layer_output_scale.weight");

        // ---- attention ----
        rmsnorm_kernel<<<B, NORM_THREADS(n_embd)>>>(dh, dx, dW_layer(m, L, "attn_norm.weight"), n_embd, eps, actq_for(B * n_embd));
        pf_tick(&g_pf_elem);

        int src = kv_src_dev(m, L);
        const int has_kv = L < c->n_kv_start;
        if (has_kv) {
            mm(dkb, wq_layer(m, L, "attn_k.weight"), dh, n_embd, kv_dim);
            const struct gguf_tensor *wv = wq_layer(m, L, "attn_v.weight");
            if (wv) mm(dvb, wv, dh, n_embd, kv_dim);
            else    CUDA_CHECK(cudaMemcpyAsync(dvb, dkb, (size_t)B * kv_dim * 4, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        }
        mm(dq, wq_layer(m, L, "attn_q.weight"), dh, n_embd, q_dim);
        pf_tick(&g_pf_mm);
        rmsnorm_kernel<<<B * n_head, 256>>>(dq, dq, dW_layer(m, L, "attn_q_norm.weight"), hd, eps, AQ0);
        rope_n_kernel<<<gridn(B * n_head * hd / 2), 256>>>(dq, hd / 2, hd, d_pos, base, ff, n_head * hd / 2, B);
        if (has_kv) {
            rmsnorm_kernel<<<B * n_head_kv, 256>>>(dkb, dkb, dW_layer(m, L, "attn_k_norm.weight"), hd, eps, AQ0);
            rmsnorm_kernel<<<B * n_head_kv, 256>>>(dvb, dvb, NULL, hd, eps, AQ0);
            rope_n_kernel<<<gridn(B * n_head_kv * hd / 2), 256>>>(dkb, hd / 2, hd, d_pos, base, ff, n_head_kv * hd / 2, B);
            if (kv->f16[L]) {                              // global layer: rows round through f16
                kv_write_h_n_kernel<<<gridn(B * kv_dim), 256>>>((__half *)kv->k[L], dkb, d_pos, kv_dim, B);
                kv_write_h_n_kernel<<<gridn(B * kv_dim), 256>>>((__half *)kv->v[L], dvb, d_pos, kv_dim, B);
            } else if (kv->seq[L] < kv->max_seq) {         // ring (sliding-window layer)
                kv_write_ring_n_kernel<<<gridn(B * kv_dim), 256>>>((float *)kv->k[L], dkb, d_pos, kv_dim, kv->seq[L], B);
                kv_write_ring_n_kernel<<<gridn(B * kv_dim), 256>>>((float *)kv->v[L], dvb, d_pos, kv_dim, kv->seq[L], B);
            } else {                                       // full-length f32: row = position
                kv_write_n_kernel<<<gridn(B * kv_dim), 256>>>((float *)kv->k[L], dkb, d_pos, kv_dim, B);
                kv_write_n_kernel<<<gridn(B * kv_dim), 256>>>((float *)kv->v[L], dvb, d_pos, kv_dim, B);
            }
        }
        const void *Kc = kv->k[src], *Vc = kv->v[src];
        pf_tick(&g_pf_elem);
        int gqa = n_head / n_head_kv;
        int window = (local && c->sliding_window > 0) ? c->sliding_window : 0;
        // K/V sharing pays only when the attended cache is too big to stay
        // L2-resident — then the per-query kernel's B× re-reads hit DRAM and
        // sharing them across the chunk's queries wins; when it fits L2 the
        // re-reads are already cheap and the sharing kernel's per-position
        // barriers only add overhead. The footprint is the K+V of the attended
        // span: full layers (window 0) grow unbounded -> always share; SWA
        // layers cap at `window` rows, so share only when window*kv_dim*KT is
        // big (true on 12B's wide kv_dim, false on E4B's narrow window). This
        // split is exactly what the Orin measured: sharing sped 12B's full AND
        // window layers but slowed E4B's window layers. The B<=2 MTP verify
        // always takes the per-query path (its argmax must match decode's).
        static long g_l2 = -1;
        if (g_l2 < 0) { cudaDeviceProp p; cudaGetDeviceProperties(&p, 0); g_l2 = p.l2CacheSize; }
        int ktsz = kv->f16[src] ? 2 : 4;
        // Tensor-core flash for the prefill chunk (B>2): ~40% of TTFT was scalar-
        // dot attention. hd 256/512 only (Gemma-4 head dims); B<=2 MTP verify and
        // other hd keep the per-query path. LG_NO_FLASH falls back. Flash writes
        // f32 xb -> act_quantize fills the int8 activation for attn_output.
        static int no_flash = -1;
        if (no_flash < 0) no_flash = getenv("LG_NO_FLASH") != NULL;
        static int no_splitk = -1;
        if (no_splitk < 0) no_splitk = getenv("LG_NO_SPLITK") != NULL;
        // LG_FORCE_KVSHARE: route E4B's flash path through K/V-sharing instead. The
        // ncu showed flash is L1-bandwidth bound (92-95%) re-reading K/V per query;
        // the L2-footprint gate that disabled sharing on E4B measured the wrong
        // resource. Test whether staging K/V to shared (reuse across QT queries)
        // relieves the L1 wall. Text-prefill only (share path has no bidir mask).
        static int force_share = -1;
        if (force_share < 0) force_share = getenv("LG_FORCE_KVSHARE") != NULL;
        bool flash = !no_flash && !force_share && B > 2 && !g_chunk_verify && (hd == 256 || hd == 512);   // y-tiled over 32-query blocks, so B>32 (128) is fine
        bool share = (B > 2) && !g_chunk_verify && (force_share || window == 0 || 2LL * window * kv_dim * ktsz > (long)g_l2);
        // The B<=2 MTP verify runs the SAME split-K kernel as decode (one extra
        // grid axis for the 2 query rows): out[0]/out[1] then byte-match plain
        // decode's forwards at pos/pos+1 by construction, and verify's attention
        // gets the split-K parallelism win at high context. Per-query stays as the
        // LG_NO_SPLITK fallback (and there decode is per-query too, so they still
        // match). Mirrors decode's gate (decode-side LG_NO_SPLITK is independent).
        bool splitk = !no_splitk && (B <= 2 || g_chunk_verify) && (hd == 256 || hd == 512);
        if (flash) {
            bool f16 = kv->f16[src], ring = (kv->seq[src] < kv->max_seq);
            if (hd == 512) launch_flash<512>(dxb, dq, Kc, Vc, kv_dim, gqa, d_pos, window, kv->seq[src], B, n_head, f16, ring, g_pf_seg, g_pf_bidir_hi);
            else           launch_flash<256>(dxb, dq, Kc, Vc, kv_dim, gqa, d_pos, window, kv->seq[src], B, n_head, f16, ring, g_pf_seg, g_pf_bidir_hi);
            act_quantize(dxb, B * q_dim);
        } else if (share) {                                // attended K/V exceeds L2: share across queries
            const int QT = 8;                              // queries per block (one warp each)
            size_t shm = (size_t)(2 + QT) * hd * sizeof(float);   // sK + sV + sO[QT][hd]
            dim3 g(n_head, (B + QT - 1) / QT);
            struct actq aq = actq_for(B * q_dim);
            if (kv->f16[src])                              // full layer, f16 global cache
                attn_kvshare_n_kernel<QT, false, __half><<<g, QT * 32, shm>>>(dxb, dq, (const __half *)Kc, (const __half *)Vc, hd, kv_dim, gqa, d_pos, window, 0, B, n_head, aq);
            else if (kv->seq[src] < kv->max_seq)           // SWA layer, ring cache (e.g. 12B's wide windows)
                attn_kvshare_n_kernel<QT, true, float><<<g, QT * 32, shm>>>(dxb, dq, (const float *)Kc, (const float *)Vc, hd, kv_dim, gqa, d_pos, window, kv->seq[src], B, n_head, aq);
            else                                           // full layer, full-length float cache
                attn_kvshare_n_kernel<QT, false, float><<<g, QT * 32, shm>>>(dxb, dq, (const float *)Kc, (const float *)Vc, hd, kv_dim, gqa, d_pos, window, 0, B, n_head, aq);
        } else if (splitk) {                               // B<=2 MTP verify: decode's split-K kernel, z=B queries
            ensure_split(B * n_head);                       // no-op (ensure_weights pre-allocated 2*n_head pre-capture)
            size_t shm = (size_t)(256 / 32) * hd * sizeof(float);
            dim3 gs(n_head, MAXSPLIT, B);
            struct actq aq = actq_for(B * q_dim);
            if (kv->f16[src])
                split_attn_kernel<false, __half><<<gs, 256, shm>>>(g_pacc, g_pml, dq, (const __half *)Kc, (const __half *)Vc, hd, kv_dim, gqa, d_pos, window, 0, n_head);
            else if (kv->seq[src] < kv->max_seq)
                split_attn_kernel<true, float><<<gs, 256, shm>>>(g_pacc, g_pml, dq, (const float *)Kc, (const float *)Vc, hd, kv_dim, gqa, d_pos, window, kv->seq[src], n_head);
            else
                split_attn_kernel<false, float><<<gs, 256, shm>>>(g_pacc, g_pml, dq, (const float *)Kc, (const float *)Vc, hd, kv_dim, gqa, d_pos, window, 0, n_head);
            combine_attn_kernel<<<dim3(n_head, 1, B), 256>>>(dxb, g_pacc, g_pml, hd, d_pos, window, aq, n_head);
        } else {
            size_t shm = (size_t)(256 / 32) * hd * sizeof(float);
            if (kv->f16[src])
                attn_h_n_kernel<<<dim3(n_head, B), 256, shm>>>(dxb, dq, (const __half *)Kc, (const __half *)Vc, hd, kv_dim, gqa, d_pos, window, actq_for(B * q_dim));
            else if (kv->seq[src] < kv->max_seq)
                attn_swa_n_kernel<<<dim3(n_head, B), 256, shm>>>(dxb, dq, (const float *)Kc, (const float *)Vc, hd, kv_dim, gqa, d_pos, window, kv->seq[src], actq_for(B * q_dim));
            else
                attn_n_kernel<<<dim3(n_head, B), 256, shm>>>(dxb, dq, (const float *)Kc, (const float *)Vc, hd, kv_dim, gqa, d_pos, window, actq_for(B * q_dim));
        }

        pf_tick(&g_pf_attn);
        mm(dout, wq_layer(m, L, "attn_output.weight"), dxb, q_dim, n_embd);
        pf_tick(&g_pf_mm);
        rmsnorm_add_n_kernel<<<B, NORM_THREADS(n_embd)>>>(dx, dout, dW_layer(m, L, "post_attention_norm.weight"), n_embd, eps, NULL, AQ0);

        // ---- feed-forward (GeGLU) ----
        const int nff = m->ffn_len[L];
        rmsnorm_kernel<<<B, NORM_THREADS(n_embd)>>>(dh, dx, dW_layer(m, L, "ffn_norm.weight"), n_embd, eps, actq_for(B * n_embd));
        pf_tick(&g_pf_elem);
        mm(dg2, wq_layer(m, L, "ffn_up.weight"), dh, n_embd, nff);
        mm(dg1, wq_layer(m, L, "ffn_gate.weight"), dh, n_embd, nff);
        pf_tick(&g_pf_mm);
        geglu_n_kernel<<<gridn(B * nff), 256>>>(dg1, dg2, nff, B, nff, actq_for(B * nff));
        pf_tick(&g_pf_elem);
        mm(dout, wq_layer(m, L, "ffn_down.weight"), dg1, nff, n_embd);
        pf_tick(&g_pf_mm);
        rmsnorm_add_n_kernel<<<B, NORM_THREADS(n_embd)>>>(dx, dout, dW_layer(m, L, "post_ffw_norm.weight"), n_embd, eps,
                                       has_ple ? NULL : os, has_ple ? actq_for(B * n_embd) : AQ0);
        pf_tick(&g_pf_elem);

        // ---- per-layer input (PLE) ----
        if (has_ple) {
            const int ple = c->n_embd_per_layer;
            mm(dpg, wq_layer(m, L, "inp_gate.weight"), dx, n_embd, ple);
            geglu_n_kernel<<<gridn(B * ple), 256>>>(dpg, d_ipl + (size_t)L * ple, ple, B, c->n_layer * ple, actq_for(B * ple));
            mm(dout, wq_layer(m, L, "proj.weight"), dpg, ple, n_embd);
            rmsnorm_add_n_kernel<<<B, NORM_THREADS(n_embd)>>>(dx, dout, dW_layer(m, L, "post_norm.weight"), n_embd, eps, os, AQ0);
            pf_tick(&g_pf_ple);
        }
    }
}

static int model_has_ple(struct model *m) {
    return m->cfg.n_embd_per_layer > 0 &&
           gguf_find_tensor(m->ctx, "per_layer_token_embd.weight") != NULL;
}

// Token form: look up and scale the chunk's embedding rows, build the PLE
// inputs, then run the layers.
static void forward_chunk(struct model *m, struct kvcache *kv, const int *tokens, int pos0, int cols) {
    const struct config *c = &m->cfg;
    g_pf_cols = cols;
    const int B = cols, n_embd = c->n_embd;

    float *rows = (float *)malloc((size_t)B * n_embd * 4);
    if (!rows) { fprintf(stderr, "forward_chunk: out of memory\n"); exit(1); }
    float es = sqrtf((float)n_embd);
    for (int j = 0; j < B; j++) {
        float *erow = dequantize_row(wq(m, "token_embd.weight"), tokens[j], n_embd);
        for (int i = 0; i < n_embd; i++) rows[(size_t)j * n_embd + i] = erow[i] * es;
        free(erow);
    }
    CUDA_CHECK(cudaMemcpy(dx, rows, (size_t)B * n_embd * 4, cudaMemcpyHostToDevice));
    free(rows);

    CUDA_CHECK(cudaMemcpy(d_pos, &pos0, sizeof(int), cudaMemcpyHostToDevice));  // kernels add the row index
    build_per_layer_n(m, tokens, B, matmul_q_n);
    chunk_layers(m, kv, model_has_ple(m), B, matmul_q_n);
}

// Embedding form (media tokens): the rows enter exactly as given — media
// embeddings are NOT sqrt(n_embd)-scaled (only real token lookups are). On a
// PLE model a media position takes the PADDING token's (id 0) per-layer row
// beside the usual projection of its embedding — the reference does exactly
// this for embedding batches; the 12B has no PLE at all.
static void forward_chunk_embd(struct model *m, struct kvcache *kv, const float *rows, int pos0, int cols) {
    const struct config *c = &m->cfg;
    g_pf_cols = cols;
    CUDA_CHECK(cudaMemcpy(dx, rows, (size_t)cols * c->n_embd * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos, &pos0, sizeof(int), cudaMemcpyHostToDevice));
    int has_ple = model_has_ple(m);
    if (has_ple) {
        int pad[PREFILL_MAX_B] = { 0 };
        build_per_layer_n(m, pad, cols, matmul_q_n);
    }
    chunk_layers(m, kv, has_ple, cols, matmul_q_n);
}

// Mixed form: each chunk position is a text token (ids[j] >= 0: looked-up,
// scaled embedding + its own per-layer row) or a media row (ids[j] < 0: row
// -ids[j]-1 of mrows as given + the padding token's per-layer row). Every
// position's math is exactly its pure-form chunk's — only the company a
// position keeps in a chunk changes.
static void forward_chunk_mixed(struct model *m, struct kvcache *kv, const float *mrows,
                                const int *ids, int pos0, int cols) {
    const struct config *c = &m->cfg;
    g_pf_cols = cols;
    const int B = cols, n_embd = c->n_embd;

    float *rows = (float *)malloc((size_t)B * n_embd * 4);
    if (!rows) { fprintf(stderr, "forward_chunk_mixed: out of memory\n"); exit(1); }
    float es = sqrtf((float)n_embd);
    int toks[PREFILL_MAX_B];
    for (int j = 0; j < B; j++) {
        if (ids[j] >= 0) {
            toks[j] = ids[j];
            float *erow = dequantize_row(wq(m, "token_embd.weight"), ids[j], n_embd);
            for (int i = 0; i < n_embd; i++) rows[(size_t)j * n_embd + i] = erow[i] * es;
            free(erow);
        } else {
            toks[j] = 0;
            memcpy(rows + (size_t)j * n_embd, mrows + (size_t)(-ids[j] - 1) * n_embd, (size_t)n_embd * 4);
        }
    }
    CUDA_CHECK(cudaMemcpy(dx, rows, (size_t)B * n_embd * 4, cudaMemcpyHostToDevice));
    free(rows);

    CUDA_CHECK(cudaMemcpy(d_pos, &pos0, sizeof(int), cudaMemcpyHostToDevice));
    build_per_layer_n(m, toks, B, matmul_q_n);
    chunk_layers(m, kv, model_has_ple(m), B, matmul_q_n);
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

extern "C" int model_forward_next(struct model *m, struct kvcache *kv, int token, int pos) {
    forward_token(m, kv, token, pos, 1);
    argmax_kernel<<<1, 1024>>>(dlogits, m->cfg.n_vocab, d_best);
    int best;
    CUDA_CHECK(cudaMemcpy(&best, d_best, sizeof(int), cudaMemcpyDeviceToHost));
    return best;
}

// Pre-size the int8 activation scratch to the 128-wide max BEFORE any chunk or
// the decode-graph capture. Adaptive chunks make a short first turn size g_xq
// small; a later wider turn would then realloc it — and the captured decode
// graph references g_xq, so a realloc would leave it dangling. One max-size call
// up front (no-op on the f32 backend) keeps the pointer stable for the session.
static void prefill_act_presize(struct model *m) {
    static int done = 0;
    if (done) return;
    actq_for((int)((size_t)g_prefill_max_b * m->cfg.n_ff));   // n_ff is the widest activation (ffn_down input)
    done = 1;
}

extern "C" void model_prefill(struct model *m, struct kvcache *kv, const int *tokens, int n, int pos0) {
    wide_chunk_init();                                // before ensure_scratch sizes buffers + ring
    ensure_weights(m);
    ensure_scratch(m);
    prefill_act_presize(m);
    const int CB = g_wide_chunk ? g_wide_chunk : PREFILL_B;
    int i = 0;
    for (; n - i >= CB; i += CB)                       // wide chunks: weights read once per CB tokens
        forward_chunk(m, kv, tokens + i, pos0 + i, CB);
    for (; n - i >= PREFILL_B; i += PREFILL_B)         // remainder (< CB) in legacy 128-chunks
        forward_chunk(m, kv, tokens + i, pos0 + i, PREFILL_B);
    // The warmup tokens exist so ensure_act's one-time cudaMallocs precede graph
    // capture; one chunk does all of them (its activations are B× decode's), so
    // skip straight to capture — otherwise a chunk-aligned prompt would push the
    // two un-captured tokens and the capture itself into the generation timer.
    if (i > 0 && g_graph_warmups < 2) g_graph_warmups = 2;
    if (pf_on() && i > 0)
        fprintf(stderr, "prefill profile (%d chunk tokens): matmul %.2fs, attention %.2fs, elementwise %.2fs, ple %.2fs\n",
                i, g_pf_mm, g_pf_attn, g_pf_elem, g_pf_ple);
    if (pf_on() && i > 0) matmul_coverage_print();
    // Remainder: PAD to one more chunk instead of single-token forwards. A
    // short serve turn (a robot camera frame, a one-line question) is MOSTLY
    // remainder, and every single costs a full decode pass — a 71-token image
    // turn spent more time in its ~20 stragglers than in its chunks. The pad
    // repeats the last real token at the following positions; those kv rows
    // are rewritten by whatever comes next (the next segment, the turn's
    // final forward) before anything can read them — every consumer writes
    // position p before its attention reads p, and reads stop at its own
    // position. Real rows' math is the chunk path's: byte-identical.
    int rem = n - i, cols = ((rem + 31) / 32) * 32;   // adaptive: round the tail up to 32, not 128
    if (cols > PREFILL_B) cols = PREFILL_B;
    if (rem >= 2 && pos0 + i + cols <= kv->max_seq) {
        int padded[PREFILL_MAX_B];
        for (int j = 0; j < cols; j++) padded[j] = tokens[i + (j < rem ? j : rem - 1)];
        forward_chunk(m, kv, padded, pos0 + i, cols);
        if (g_graph_warmups < 2) g_graph_warmups = 2;
        return;
    }
    for (; i < n; i++)                                // 0-1 tokens (or no room): singles
        forward_token(m, kv, tokens[i], pos0 + i, 0);
}

extern "C" void model_prefill_embd(struct model *m, struct kvcache *kv, const float *rows, int n, int pos0) {
    ensure_weights(m);
    ensure_scratch(m);
    prefill_act_presize(m);
    const int n_embd = m->cfg.n_embd;
    int i = 0;
    for (; n - i >= PREFILL_B; i += PREFILL_B)
        forward_chunk_embd(m, kv, rows + (size_t)i * n_embd, pos0 + i, PREFILL_B);
    if (i > 0 && g_graph_warmups < 2) g_graph_warmups = 2;
    // pad the remainder to an adaptive-width chunk (roundup to 32, not 128); the
    // last row repeats; the pad rows' kv is overwritten before it can be read.
    int rem = n - i, cols = ((rem + 31) / 32) * 32;
    if (cols > PREFILL_B) cols = PREFILL_B;
    if (rem >= 2 && pos0 + i + cols <= kv->max_seq) {
        float *padded = (float *)malloc((size_t)cols * n_embd * 4);
        if (padded) {
            for (int j = 0; j < cols; j++)
                memcpy(padded + (size_t)j * n_embd,
                       rows + (size_t)(i + (j < rem ? j : rem - 1)) * n_embd, (size_t)n_embd * 4);
            forward_chunk_embd(m, kv, padded, pos0 + i, cols);
            free(padded);
            if (g_graph_warmups < 2) g_graph_warmups = 2;
            return;
        }
    }
    for (; i < n; i++) {                              // 0-1 rows (or no room): singles
        CUDA_CHECK(cudaMemcpy(dx, rows + (size_t)i * n_embd, (size_t)n_embd * 4, cudaMemcpyHostToDevice));
        int pos = pos0 + i;
        CUDA_CHECK(cudaMemcpy(d_pos, &pos, sizeof(int), cudaMemcpyHostToDevice));
        if (model_has_ple(m)) build_per_layer(m, 0);  // padding token's PLE row
        forward_graph(m, kv, 0);
    }
}

extern "C" void model_prefill_mixed(struct model *m, struct kvcache *kv, const float *rows,
                                    const int *ids, int n, int pos0) {
    ensure_weights(m);
    ensure_scratch(m);
    prefill_act_presize(m);
    const int n_embd = m->cfg.n_embd;

    // Per-position media-span ids: text -> 0, a media token -> its span's start abs
    // position+1 (a unique nonzero). Uploaded once so the flash mask can attend
    // bidirectionally within a frame (Gemma's image/video token-type behaviour).
    if (kv->max_seq > g_seg_cap) {
        if (g_seg_dev) cudaFree(g_seg_dev);
        CUDA_CHECK(cudaMalloc(&g_seg_dev, (size_t)kv->max_seq * sizeof(int)));
        CUDA_CHECK(cudaMemset(g_seg_dev, 0, (size_t)kv->max_seq * sizeof(int)));
        g_seg_cap = kv->max_seq;
    }
    int seg_ok = (pos0 + n <= kv->max_seq);
    if (seg_ok) {
        int *hseg = (int *)malloc((size_t)n * sizeof(int));
        int run0 = 0;
        for (int j = 0; j < n; j++) {
            if (ids[j] < 0) { if (j == 0 || ids[j - 1] >= 0) run0 = pos0 + j + 1; hseg[j] = run0; }
            else hseg[j] = 0;
        }
        CUDA_CHECK(cudaMemcpy(g_seg_dev + pos0, hseg, (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
        free(hseg);
    }

    // Segment-aware chunking: pack greedily to PREFILL_B, but never split a media
    // span across chunks (a frame's patches must coexist in one attention pass for
    // bidirectional-within-frame). A span larger than a chunk falls back to causal
    // sub-chunks (Stage 2 will widen the chunk for it). Text-only chunks pass
    // seg=NULL -> byte-identical causal; chunking matches the old 128 path for text.
    int i = 0;
    while (n - i >= 2) {
        int a = i, w = 0, media_hi = -1;
        // A whole image span prefills as ONE chunk so its patches attend
        // bidirectionally. If the span exceeds PREFILL_B it becomes its OWN wide
        // chunk (up to g_prefill_max_b, the buffer width); otherwise pack text +
        // small spans greedily to PREFILL_B (causal text, bidirectional frames).
        if (ids[a] < 0) {
            int e = a; while (e < n && ids[e] < 0) e++;
            int span = e - a;
            if (span > PREFILL_B) { w = (span <= g_prefill_max_b) ? span : g_prefill_max_b; media_hi = a + w - 1; }
        }
        if (w == 0) while (a + w < n && w < PREFILL_B) {
            if (ids[a + w] < 0) {                      // a media span: take it whole if it fits
                int s = a + w, e = s; while (e < n && ids[e] < 0) e++;
                int span = e - s;
                if (span > PREFILL_B) break;           // wide span -> close chunk, it starts the next (its own wide chunk)
                if (w + span > PREFILL_B) break;       // doesn't fit the remaining budget -> close chunk
                media_hi = e - 1; w += span;
            } else w++;                                // text
        }
        if (w < 1) w = 1;
        int cols = ((w + 31) / 32) * 32; if (cols > g_prefill_max_b) cols = g_prefill_max_b;
        if (pos0 + a + cols > kv->max_seq) break;
        int padded[PREFILL_MAX_B];
        for (int j = 0; j < cols; j++) padded[j] = ids[a + (j < w ? j : w - 1)];
        static int no_bidir = -1; if (no_bidir < 0) no_bidir = getenv("LG_NO_IMG_BIDIR") != NULL;
        int bidir = seg_ok && media_hi >= 0 && !no_bidir;
        g_pf_seg = bidir ? g_seg_dev : NULL;
        g_pf_bidir_hi = bidir ? (pos0 + media_hi) : 0;
        forward_chunk_mixed(m, kv, rows, padded, pos0 + a, cols);
        if (g_graph_warmups < 2) g_graph_warmups = 2;
        i = a + w;
    }
    g_pf_seg = NULL; g_pf_bidir_hi = 0;
    for (; i < n; i++) {                              // 0-1 trailing positions: singles
        if (ids[i] >= 0) { forward_token(m, kv, ids[i], pos0 + i, 0); continue; }
        CUDA_CHECK(cudaMemcpy(dx, rows + (size_t)(-ids[i] - 1) * n_embd, (size_t)n_embd * 4, cudaMemcpyHostToDevice));
        int pos = pos0 + i;
        CUDA_CHECK(cudaMemcpy(d_pos, &pos, sizeof(int), cudaMemcpyHostToDevice));
        if (model_has_ple(m)) build_per_layer(m, 0);  // padding token's PLE row
        forward_graph(m, kv, 0);
    }
}

// The MTP verify step: the LG_MTP_N tokens (toks[0]=chosen, toks[1..]=drafts) as ONE
// B=LG_MTP_N chunk — each weight matrix crosses memory once for the whole block, the
// entire economics of speculative decoding on a bandwidth-bound device. The head runs
// at ALL rows (the only place that happens; prefill always skips it). out[j] = greedy
// successor of toks[j]; out[0] always valid, out[j] valid only when every earlier
// draft held. g_hidden is left at the last VALID row's post-norm hidden so the next
// draft chains on accept and reject alike; rejected rows are overwritten before any
// query reads them (causality, no rollback). Everything reads d_pos on-device, so node
// parameters never vary -> captures into a static CUDA graph like decode (the
// un-captured form is ~1030 raw launches/step, where sync MTP died on WDDM).
static void verify_head_spec(struct model *m) {
    const struct config *c = &m->cfg;
    const int n_embd = c->n_embd, N = LG_MTP_N;
    rmsnorm_kernel<<<N, NORM_THREADS(n_embd)>>>(dx, dx, dW(m, "output_norm.weight"), n_embd, c->rms_eps, actq_for(N * n_embd));
    matmul_q_spec(d_logits_spec, wq(m, "token_embd.weight"), dx, n_embd, c->n_vocab);
    if (c->logit_softcap > 0.0f)
        softcap_kernel<<<gridn(N * c->n_vocab), 256>>>(d_logits_spec, c->logit_softcap, N * c->n_vocab);
    for (int j = 0; j < N; j++)
        argmax_kernel<<<1, 1024>>>(d_logits_spec + (size_t)j * c->n_vocab, c->n_vocab, d_best_spec + j);
}
static void verify_layers_and_head_spec(struct model *m, struct kvcache *kv, int has_ple) {
    g_chunk_verify = 1;                  // verify uses decode's split-K attn, not prefill flash/share
    chunk_layers(m, kv, has_ple, LG_MTP_N, matmul_q_spec);
    g_chunk_verify = 0;
    verify_head_spec(m);
}

static double now_sec_dev(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

// LG_MTP_PROFILE=1: run the verify un-captured with syncs around its two
// halves and report — the tool that finds where a 2x-over-theory round goes.
static void verify_profiled(struct model *m, struct kvcache *kv, int has_ple) {
    static double tl = 0, th = 0;
    static int n = 0;
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    double t0 = now_sec_dev();
    g_chunk_verify = 1;
    chunk_layers(m, kv, has_ple, LG_MTP_N, matmul_q_spec);
    g_chunk_verify = 0;
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    double t1 = now_sec_dev();
    verify_head_spec(m);
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    tl += t1 - t0; th += now_sec_dev() - t1;
    if (++n % 50 == 0)
        fprintf(stderr, "verify profile over %d: layers %.1fms ea, head %.1fms ea\n", n, 1e3 * tl / n, 1e3 * th / n);
}

static cudaGraphExec_t g_graph_spec_exec = NULL;
static int g_graph_spec_warmups = 0;
static void verify_graph_spec(struct model *m, struct kvcache *kv, int has_ple) {
    if (g_graph_spec_warmups < 2) {              // one-time mallocs (ensure_act) finish un-captured
        g_graph_spec_warmups++;
        verify_layers_and_head_spec(m, kv, has_ple);
        return;
    }
    if (!g_graph_spec_exec) {
        cudaGraph_t graph;
        CUDA_CHECK(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        verify_layers_and_head_spec(m, kv, has_ple);
        CUDA_CHECK(cudaStreamEndCapture(cudaStreamPerThread, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&g_graph_spec_exec, graph, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    CUDA_CHECK(cudaGraphLaunch(g_graph_spec_exec, cudaStreamPerThread));
}

extern "C" int model_forward_spec(struct model *m, struct kvcache *kv, const int *toks, int pos, int *out) {
    const struct config *c = &m->cfg;
    const int n_embd = c->n_embd, N = LG_MTP_N;
    float *rows = (float *)malloc((size_t)N * n_embd * 4);
    if (!rows) { fprintf(stderr, "model_forward_spec: out of memory\n"); exit(1); }
    float es = sqrtf((float)n_embd);
    for (int j = 0; j < N; j++) {
        float *erow = dequantize_row(wq(m, "token_embd.weight"), toks[j], n_embd);
        for (int i = 0; i < n_embd; i++) rows[(size_t)j * n_embd + i] = erow[i] * es;
        free(erow);
    }
    CUDA_CHECK(cudaMemcpy(dx, rows, (size_t)N * n_embd * 4, cudaMemcpyHostToDevice));
    free(rows);
    CUDA_CHECK(cudaMemcpy(d_pos, &pos, sizeof(int), cudaMemcpyHostToDevice));

    int has_ple = model_has_ple(m);
    if (has_ple) build_per_layer_n(m, toks, N, matmul_q_spec);   // un-captured, like decode's PLE build
    static int prof = -1;
    if (prof < 0) prof = getenv("LG_MTP_PROFILE") != NULL;
    if (prof) verify_profiled(m, kv, has_ple);
    else      verify_graph_spec(m, kv, has_ple);
    CUDA_CHECK(cudaMemcpy(out, d_best_spec, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost));

    // accept run: out[0] always valid; out[j] valid iff every earlier draft held.
    int adv = 1;
    while (adv < N && out[adv - 1] == toks[adv]) adv++;
    CUDA_CHECK(cudaMemcpyAsync(g_hidden, dx + (size_t)(adv - 1) * n_embd, (size_t)n_embd * 4,
                               cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    return adv;
}

// ==== MTP: the draft head, on the device =====================================
// The gemma4-assistant forward from src/mtp.c as ~30 kernel launches per
// draft. Almost everything is REUSE: rmsnorm_kernel (pre/post norms and, with
// grid = heads, the per-head q norm — exactly how decode uses it),
// geglu_kernel, softcap_kernel, argmax_kernel, and above all the d_attn
// template: a draft for position `pos` sees the cache exactly as a query at
// pos-1 with the window shrunk by one — [pos-window+1, pos-1] sliding,
// [0, pos-1] full — so two NEW wrappers below pin those arguments and the
// frozen decode entry points stay untouched. Assistant weights are dequantized
// once to f16 on the device (77M params, ~150 MB); h_prev is g_hidden, which
// the decode graph already maintains. The whole draft CAPTURES into one CUDA
// graph (position read on-device from d_dpos, like decode's d_pos), so a
// round costs one graph launch + one 4-byte readback instead of ~30 raw
// launches — on WDDM that launch tax was most of the draft's wall time.

__global__ static void mtp_matvec_h(float *out, const __half *W, const float *x, int k, int m) {
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= m) return;
    const __half2 *w2 = (const __half2 *)(const void *)(W + (size_t)row * k);
    float s = 0.0f;
    for (int i = lane; i < k / 2; i += 32) {
        float2 wf = __half22float2(w2[i]);
        s += wf.x * x[2 * i] + wf.y * x[2 * i + 1];
    }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if (!lane) out[row] = s;
}

// NeoX rope at the device-resident draft position (static graph node). No
// freq factors: the assistant has none.
__global__ static void mtp_rope(float *v, int half, int hd, const int *dp, float base, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int pos = *dp;
    int head = idx / half, i = idx % half;
    float *vh = v + (size_t)head * hd;
    float freq = powf(base, -2.0f * (float)i / (float)hd);
    float ang = (float)pos * freq;
    float c = cosf(ang), s = sinf(ang), a = vh[i], b = vh[i + half];
    vh[i] = a * c - b * s; vh[i + half] = a * s + b * c;
}

__global__ static void mtp_add_scale(float *x, const float *o, int n, float sc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = (x[i] + o[i]) * sc;
}

// The draft at *dp sees the cache as a query at *dp - 1 (see the host draft).
__global__ static void mtp_attn_h(float *xb, const float *q, const __half *Kc, const __half *Vc,
                                  int hd, int kv_dim, int gqa, const int *dp, struct actq aq) {
    d_attn<false, __half>(xb, q, Kc, Vc, hd, kv_dim, gqa, *dp - 1, 0, 0, 0, aq);
}
__global__ static void mtp_attn_swa(float *xb, const float *q, const float *Kc, const float *Vc,
                                    int hd, int kv_dim, int gqa, const int *dp, int window, int seq, struct actq aq) {
    d_attn<true, float>(xb, q, Kc, Vc, hd, kv_dim, gqa, *dp - 1, 0, window, seq, aq);
}

struct mtp_ld {
    float *attn_norm, *q_norm, *post_attn, *ffn_norm, *post_ffw;   // f32, device
    float out_scale;                                               // scalar, by value
    __half *q, *o, *gate, *up, *down;
};
struct mtp_cuda {
    struct mtp_ld *l;
    __half *pre, *post, *head;
    float *out_norm;
    float *cat, *x, *h, *q, *xb, *o, *g1, *g2, *logits;            // device scratch
    int *d_tok;
    int *d_dpos;                  // device-resident draft position (static graph)
    cudaGraphExec_t graph;
    int warmups;
};

static __half *mtp_up_h(const struct gguf_tensor *t) {            // any type -> device f16
    size_t n = 1;
    for (uint32_t i = 0; i < t->n_dims; i++) n *= t->dims[i];
    float *f = (float *)malloc(n * 4);
    __half *hh = (__half *)malloc(n * 2);
    __half *d = NULL;
    int ok = f && hh && dequantize_into(t->type, t->data, f, (int64_t)n) &&
             cudaMalloc(&d, n * 2) == cudaSuccess;
    if (ok) {
        for (size_t i = 0; i < n; i++) hh[i] = __float2half(f[i]);
        ok = cudaMemcpy(d, hh, n * 2, cudaMemcpyHostToDevice) == cudaSuccess;
    }
    free(f); free(hh);
    return ok ? d : NULL;
}
static float *mtp_up_f(const float *src, size_t n) {
    float *d = NULL;
    if (cudaMalloc(&d, n * 4) != cudaSuccess) return NULL;
    return cudaMemcpy(d, src, n * 4, cudaMemcpyHostToDevice) == cudaSuccess ? d : NULL;
}

static struct mtp_cuda *mtp_cuda_init(struct mtp *t) {
    struct mtp_cuda *mc = (struct mtp_cuda *)calloc(1, sizeof *mc);
    if (!mc) return NULL;
    mc->l = (struct mtp_ld *)calloc((size_t)t->n_layer, sizeof *mc->l);
    mc->pre  = mtp_up_h(t->pre);
    mc->head = mtp_up_h(t->head);
    mc->out_norm = mtp_up_f(t->out_norm, (size_t)t->n_inner);
    // post-projection (next-token), ni->nb: lifts the head's own hidden back to
    // model width to chain a 2nd draft (LG_MTP_N=3 block-3). Optional — left NULL
    // (block-3 chaining disabled) if it's absent or not the expected ni->nb shape.
    if (t->post && t->post->n_dims == 2 &&
        (int)t->post->dims[0] == t->n_inner && (int)t->post->dims[1] == t->n_bb)
        mc->post = mtp_up_h(t->post);
    int ok = mc->l && mc->pre && mc->head && mc->out_norm;
    int q_max = 0;
    for (int L = 0; ok && L < t->n_layer; L++) {
        const struct mtp_layer *s = &t->l[L];
        struct mtp_ld *d = &mc->l[L];
        if (t->n_head * s->hd > q_max) q_max = t->n_head * s->hd;
        d->out_scale = s->out_scale ? s->out_scale[0] : 1.0f;
        ok = (d->attn_norm = mtp_up_f(s->attn_norm, t->n_inner)) &&
             (d->q_norm = mtp_up_f(s->q_norm, s->hd)) &&
             (d->post_attn = mtp_up_f(s->post_attn, t->n_inner)) &&
             (d->ffn_norm = mtp_up_f(s->ffn_norm, t->n_inner)) &&
             (d->post_ffw = mtp_up_f(s->post_ffw, t->n_inner)) &&
             (d->q = mtp_up_h(s->q)) && (d->o = mtp_up_h(s->o)) &&
             (d->gate = mtp_up_h(s->gate)) && (d->up = mtp_up_h(s->up)) &&
             (d->down = mtp_up_h(s->down)) != NULL;
    }
    ok = ok && cudaMalloc(&mc->cat, (size_t)2 * t->n_bb * 4) == cudaSuccess
            && cudaMalloc(&mc->x, (size_t)t->n_inner * 4) == cudaSuccess
            && cudaMalloc(&mc->h, (size_t)t->n_inner * 4) == cudaSuccess
            && cudaMalloc(&mc->q, (size_t)q_max * 4) == cudaSuccess
            && cudaMalloc(&mc->xb, (size_t)q_max * 4) == cudaSuccess
            && cudaMalloc(&mc->o, (size_t)t->n_inner * 4) == cudaSuccess
            && cudaMalloc(&mc->g1, (size_t)t->n_ff * 4) == cudaSuccess
            && cudaMalloc(&mc->g2, (size_t)t->n_ff * 4) == cudaSuccess
            && cudaMalloc(&mc->logits, (size_t)t->n_vocab * 4) == cudaSuccess
            && cudaMalloc(&mc->d_tok, sizeof(int)) == cudaSuccess
            && cudaMalloc(&mc->d_dpos, sizeof(int)) == cudaSuccess;
    if (!ok) {                                       // startup-OOM only; partial uploads leak, like media-cuda
        fprintf(stderr, "mtp: device upload failed, drafting disabled\n");
        free(mc->l); free(mc);
        return NULL;
    }
    return mc;
}

// Every launch in the draft, position read from mc->d_dpos — capturable.
static void mtp_draft_launches(struct mtp *t, const struct model *m, const struct kvcache *kv,
                               struct mtp_cuda *mc) {
    const struct config *c = &m->cfg;
    const int ni = t->n_inner, nb = t->n_bb, nff = t->n_ff;
    const float eps = t->eps;

    // cat (= [sqrt-scaled embed(token), h_prev]) is filled by the caller BEFORE
    // this runs, so the captured graph is identical for draft 1 (h_prev=g_hidden)
    // and the chained draft 2 (h_prev=post(head hidden)).
    mtp_matvec_h<<<gridn(ni * 32), 256>>>(mc->x, mc->pre, mc->cat, 2 * nb, ni);

    for (int L = 0; L < t->n_layer; L++) {
        const struct mtp_layer *bl = &t->l[L];
        const struct mtp_ld *dl = &mc->l[L];
        const int hd = bl->hd, src = bl->src, q_dim = t->n_head * hd;
        const int gqa = t->n_head / m->n_head_kv[src];

        rmsnorm_kernel<<<1, NORM_THREADS(ni)>>>(mc->h, mc->x, dl->attn_norm, ni, eps, AQ0);
        mtp_matvec_h<<<gridn(q_dim * 32), 256>>>(mc->q, dl->q, mc->h, ni, q_dim);
        rmsnorm_kernel<<<t->n_head, 256>>>(mc->q, mc->q, dl->q_norm, hd, eps, AQ0);
        mtp_rope<<<gridn(t->n_head * (hd / 2)), 256>>>(mc->q, hd / 2, hd, mc->d_dpos,
                                                       bl->local ? t->base_swa : t->base_full,
                                                       t->n_head * (hd / 2));
        if (bl->local)
            mtp_attn_swa<<<t->n_head, 256, 8 * hd * 4>>>(mc->xb, mc->q,
                    (const float *)kv->k[src], (const float *)kv->v[src],
                    hd, kv->kv_dim[src], gqa, mc->d_dpos,
                    c->sliding_window > 0 ? c->sliding_window - 1 : 0, kv->seq[src], AQ0);
        else
            mtp_attn_h<<<t->n_head, 256, 8 * hd * 4>>>(mc->xb, mc->q,
                    (const __half *)kv->k[src], (const __half *)kv->v[src],
                    hd, kv->kv_dim[src], gqa, mc->d_dpos, AQ0);
        mtp_matvec_h<<<gridn(ni * 32), 256>>>(mc->o, dl->o, mc->xb, q_dim, ni);
        rmsnorm_kernel<<<1, NORM_THREADS(ni)>>>(mc->o, mc->o, dl->post_attn, ni, eps, AQ0);
        mtp_add_scale<<<gridn(ni), 256>>>(mc->x, mc->o, ni, 1.0f);

        rmsnorm_kernel<<<1, NORM_THREADS(ni)>>>(mc->h, mc->x, dl->ffn_norm, ni, eps, AQ0);
        mtp_matvec_h<<<gridn(nff * 32), 256>>>(mc->g1, dl->gate, mc->h, ni, nff);
        mtp_matvec_h<<<gridn(nff * 32), 256>>>(mc->g2, dl->up, mc->h, ni, nff);
        geglu_kernel<<<gridn(nff), 256>>>(mc->g1, mc->g2, nff, AQ0);
        mtp_matvec_h<<<gridn(ni * 32), 256>>>(mc->o, dl->down, mc->g1, nff, ni);
        rmsnorm_kernel<<<1, NORM_THREADS(ni)>>>(mc->o, mc->o, dl->post_ffw, ni, eps, AQ0);
        mtp_add_scale<<<gridn(ni), 256>>>(mc->x, mc->o, ni, dl->out_scale);
    }

    rmsnorm_kernel<<<1, NORM_THREADS(ni)>>>(mc->x, mc->x, mc->out_norm, ni, eps, AQ0);
    mtp_matvec_h<<<gridn(t->n_vocab * 32), 256>>>(mc->logits, mc->head, mc->x, ni, t->n_vocab);
    if (t->softcap > 0.0f)
        softcap_kernel<<<gridn(t->n_vocab), 256>>>(mc->logits, t->softcap, t->n_vocab);
    argmax_kernel<<<1, 1024>>>(mc->logits, t->n_vocab, mc->d_tok);
}

extern "C" int mtp_draft_device(struct mtp *t, const struct model *m, const struct kvcache *kv,
                                int token, int pos) {
    if (t->cuda == (void *)-1) return -1;
    struct mtp_cuda *mc = (struct mtp_cuda *)t->cuda;
    if (!mc) {
        mc = mtp_cuda_init(t);
        t->cuda = mc ? (void *)mc : (void *)-1;
        if (!mc) return -1;
    }
    const int nb = t->n_bb;

    // uploads stay outside the graph (fixed buffers, varying data)
    float *erow = dequantize_row(gguf_find_tensor(m->ctx, "token_embd.weight"), token, nb);
    if (!erow) return -1;
    float sc = sqrtf((float)nb);
    for (int i = 0; i < nb; i++) erow[i] *= sc;
    CUDA_CHECK(cudaMemcpy(mc->cat, erow, (size_t)nb * 4, cudaMemcpyHostToDevice));
    free(erow);
    CUDA_CHECK(cudaMemcpy(mc->d_dpos, &pos, sizeof(int), cudaMemcpyHostToDevice));
    // draft 1 chains on the TARGET's hidden (g_hidden); filled here, outside the
    // captured head-pass graph, so the chained draft 2 can reuse the same graph.
    CUDA_CHECK(cudaMemcpyAsync(mc->cat + nb, g_hidden, (size_t)nb * 4, cudaMemcpyDeviceToDevice, cudaStreamPerThread));

    if (mc->warmups < 2) {
        mc->warmups++;
        mtp_draft_launches(t, m, kv, mc);
    } else {
        if (!mc->graph) {
            cudaGraph_t graph;
            CUDA_CHECK(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            mtp_draft_launches(t, m, kv, mc);
            CUDA_CHECK(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&mc->graph, graph, 0));
            CUDA_CHECK(cudaGraphDestroy(graph));
        }
        CUDA_CHECK(cudaGraphLaunch(mc->graph, cudaStreamPerThread));
    }

    int best = -1;
    CUDA_CHECK(cudaMemcpy(&best, mc->d_tok, sizeof(int), cudaMemcpyDeviceToHost));
    return best;
}

// The chained draft for block-3 (LG_MTP_N=3): draft the token after `token` (itself
// the previous draft, sitting at `pos`-1) using the head's OWN hidden from the prior
// draft (mc->x, lifted ni->nb by the post-projection) as h_prev — the target never
// ran this position, so its hidden isn't available, which is why a chained draft is
// inherently weaker. Reuses draft 1's captured head-pass graph; cat is filled here
// before the launch. Returns -1 if the post-projection is unavailable (no chaining).
extern "C" int mtp_draft_chain_device(struct mtp *t, const struct model *m, const struct kvcache *kv,
                                      int token, int pos) {
    if (t->cuda == (void *)-1) return -1;
    struct mtp_cuda *mc = (struct mtp_cuda *)t->cuda;
    if (!mc || !mc->post) return -1;
    const int nb = t->n_bb, ni = t->n_inner;

    float *erow = dequantize_row(gguf_find_tensor(m->ctx, "token_embd.weight"), token, nb);
    if (!erow) return -1;
    float sc = sqrtf((float)nb);
    for (int i = 0; i < nb; i++) erow[i] *= sc;
    CUDA_CHECK(cudaMemcpy(mc->cat, erow, (size_t)nb * 4, cudaMemcpyHostToDevice));
    free(erow);
    // h_prev = post(previous draft's head hidden in mc->x); runs before the graph's
    // pre-projection overwrites mc->x (same stream -> ordered).
    mtp_matvec_h<<<gridn(nb * 32), 256>>>(mc->cat + nb, mc->post, mc->x, ni, nb);
    CUDA_CHECK(cudaMemcpy(mc->d_dpos, &pos, sizeof(int), cudaMemcpyHostToDevice));

    if (mc->graph) CUDA_CHECK(cudaGraphLaunch(mc->graph, cudaStreamPerThread));
    else           mtp_draft_launches(t, m, kv, mc);   // graph not captured yet (warmup rounds)

    int best = -1;
    CUDA_CHECK(cudaMemcpy(&best, mc->d_tok, sizeof(int), cudaMemcpyDeviceToHost));
    return best;
}

extern "C" void mtp_free_device(struct mtp *t) {
    struct mtp_cuda *mc = (struct mtp_cuda *)t->cuda;
    if (!mc || t->cuda == (void *)-1) return;
    for (int L = 0; L < t->n_layer; L++) {
        struct mtp_ld *d = &mc->l[L];
        cudaFree(d->attn_norm); cudaFree(d->q_norm); cudaFree(d->post_attn);
        cudaFree(d->ffn_norm); cudaFree(d->post_ffw);
        cudaFree(d->q); cudaFree(d->o); cudaFree(d->gate); cudaFree(d->up); cudaFree(d->down);
    }
    cudaFree(mc->pre); cudaFree(mc->post); cudaFree(mc->head); cudaFree(mc->out_norm);
    cudaFree(mc->cat); cudaFree(mc->x); cudaFree(mc->h); cudaFree(mc->q); cudaFree(mc->xb);
    cudaFree(mc->o); cudaFree(mc->g1); cudaFree(mc->g2); cudaFree(mc->logits); cudaFree(mc->d_tok);
    cudaFree(mc->d_dpos);
    if (mc->graph) cudaGraphExecDestroy(mc->graph);
    free(mc->l);
    free(mc);
    t->cuda = NULL;
}

#endif // MODEL_CUDA_CUH
