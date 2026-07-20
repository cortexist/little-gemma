// GPU gemma4v encoder — the same 16-block vision transformer media.c runs on
// the host, as a handful of CUDA kernels. The host path stays in the binary
// as oracle (LG_MEDIA_VERIFY=1) and fallback; this file exists because even a
// well-vectorized CPU needs ~15 s per 266-token image for the ~1 TFLOP this
// costs.
//
// The heavy math (GEMMs and attention, ~97% of the frame) runs on tensor
// cores: m16n8k16, f16 inputs, f32 accumulation. The weights are f16 in the
// file and stay f16 into the mma; activations live in f32 between kernels and
// are rounded to f16 at the GEMM and attention seams. That input rounding is
// the whole numerics story — LG_MEDIA_VERIFY reports ~1e-3 vs the host's f32,
// the precision class of the LLM's f16 flash path (and tighter than
// llama.cpp's f16-ACCUMULATE cuBLAS encoder). Everything between the matmuls
// (norms, rope, residuals) stays f32; default stream, no CUDA graph — one
// image is a few hundred launches and the tensor cores, not the launch gaps,
// are the story. The first cut of this file was all-f32 CUDA-core FMA,
// readable but ~5x off: the Orin spent longer in the encoder than in the LLM
// it feeds, which is backwards for a camera.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <cuda_fp16.h>

extern "C" {                     // the C headers carry no C++ linkage guards
#include "gguf.h"
#include "quant.h"
#include "media-internal.h"
}

#define VC_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "media-kernel: %s -> %s\n", #x, cudaGetErrorString(e_)); return NULL; } } while (0)

// Device-side mirror of struct vlayer.
struct vld {
    float *ln1, *ln2, *attn_post, *ffn_post, *qn, *kn;
    __half *q, *k, *v, *o, *gate, *up, *down;
};

struct vcuda {
    struct vld *vl;                 // host array of device pointers
    __half *patch16, *mmv;
    float *vpos;                    // [2*pos_size][768] learned x/y position tables
    int np_cap;                     // current size of the work buffers, in patches
    uint8_t *rgb;                   // the raw frame, as received
    float *F;                       // [np][768] im2col patch matrix
    float *X, *H, *Q, *K, *V, *D;   // [np][768]
    float *G, *U;                   // [np][3072]
    float *pooled, *rows;           // [np/9][768], [np/9][n_embd]
};

// ---- kernels ----------------------------------------------------------------

// The m16n8k16 idiom, same as the LLM's flash kernel (fragment layout pinned
// by .scratch/mma6_test.cu): a-frags are two packed halves per register from
// the 16x16 row-major A tile, b-frags two packed halves per register from the
// 16x8 col-major B tile — and "col-major B" is exactly a row of W[n][k] (or a
// key row, or a transposed V column), so every fragment here is one aligned
// 4-byte shared read.
static __device__ __forceinline__ uint32_t vh_pk(__half a, __half b) {
    return (uint32_t)__half_as_ushort(a) | ((uint32_t)__half_as_ushort(b) << 16);
}
static __device__ __forceinline__ uint32_t vh_pkf(float a, float b) {
    return vh_pk(__float2half(a), __float2half(b));
}
static __device__ __forceinline__ void vh_mma(float *c,
        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// C[t][n] = A[t][k] . W[n][k] — tensor-core GEMM. A (f32) is rounded to f16
// when staged; W stays the f16 it is in memory (the f32-FMA kernel this
// replaced widened every weight element to f32 per use and left the tensor
// cores idle — ~4x the silicon for this shape). 256 threads = 8 warps, 64x64
// output tile, each warp an m16xn32 stripe (4 n8 mma tiles sharing its
// a-frags); k staged in 64-deep shared slabs. K must be even; T, N arbitrary
// (short/odd edges are zero-padded in staging and masked at the write).
#define GK 64                   // k-slab depth
static __global__ void k_gemm(float *C, const float *A, const __half *W, int T, int K, int N) {
    __shared__ __half a[64][GK + 8], w[64][GK + 8];     // +8: 4B-aligned pairs, bank-spread rows
    int t0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    int warp = threadIdx.x >> 5, g = (threadIdx.x & 31) >> 2, tig = threadIdx.x & 3;
    int ar = (warp & 3) * 16, wr = (warp >> 2) * 32;    // warp's rows in a[], w[]
    float acc[4][4] = { 0 };
    for (int k0 = 0; k0 < K; k0 += GK) {
        for (int e = threadIdx.x; e < 64 * GK / 2; e += 256) {  // stage pairs, coalesced
            int r = e / (GK / 2), i = e % (GK / 2) * 2, ka = k0 + i;
            float2 av = t0 + r < T && ka < K ? *(const float2 *)&A[(size_t)(t0 + r) * K + ka]
                                             : make_float2(0.0f, 0.0f);
            *(uint32_t *)&a[r][i] = vh_pkf(av.x, av.y);
            *(uint32_t *)&w[r][i] = n0 + r < N && ka < K ? *(const uint32_t *)&W[(size_t)(n0 + r) * K + ka]
                                                         : 0u;
        }
        __syncthreads();
        for (int ks = 0; ks < GK; ks += 16) {
            uint32_t a0 = *(const uint32_t *)&a[ar + g][ks + tig * 2];
            uint32_t a1 = *(const uint32_t *)&a[ar + g + 8][ks + tig * 2];
            uint32_t a2 = *(const uint32_t *)&a[ar + g][ks + tig * 2 + 8];
            uint32_t a3 = *(const uint32_t *)&a[ar + g + 8][ks + tig * 2 + 8];
            for (int nt = 0; nt < 4; nt++)
                vh_mma(acc[nt],
                       a0, a1, a2, a3,
                       *(const uint32_t *)&w[wr + nt * 8 + g][ks + tig * 2],
                       *(const uint32_t *)&w[wr + nt * 8 + g][ks + tig * 2 + 8]);
        }
        __syncthreads();
    }
    for (int nt = 0; nt < 4; nt++) {
        int tt = t0 + ar + g, tn = n0 + wr + nt * 8 + tig * 2;
        if (tn >= N) continue;                          // N even => the pair is in or out together
        if (tt < T)     { C[(size_t)tt * N + tn] = acc[nt][0];       C[(size_t)tt * N + tn + 1] = acc[nt][1]; }
        if (tt + 8 < T) { C[(size_t)(tt + 8) * N + tn] = acc[nt][2]; C[(size_t)(tt + 8) * N + tn + 1] = acc[nt][3]; }
    }
}

// out[row] = rms_norm(in[row]) * w   (w NULL = plain norm). One block per row.
static __global__ void k_rms(float *out, const float *in, const float *w, int n, float eps) {
    __shared__ float red[256];
    const float *x = in + (size_t)blockIdx.x * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += 256) ss += x[i] * x[i];
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float sc = rsqrtf(red[0] / (float)n + eps);
    float *y = out + (size_t)blockIdx.x * n;
    for (int i = threadIdx.x; i < n; i += 256) y[i] = x[i] * sc * (w ? w[i] : 1.0f);
}

static __global__ void k_add(float *x, const float *a, size_t total) {
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < total; i += gridDim.x * 256ull) x[i] += a[i];
}

static __global__ void k_gelu_mul(float *g, const float *u, size_t total) {
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < total; i += gridDim.x * 256ull) {
        float x = g[i];
        g[i] = x / (1.0f + __expf(-1.702f * x)) * u[i];
    }
}

// rms-norm one 64-wide head segment with a warp (2 lanes-worth of elements each).
static __device__ void d_rms64(float *s, const float *w, float eps, int lane) {
    float ss = 0.0f;
    for (int i = lane; i < 64; i += 32) ss += s[i] * s[i];
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    float sc = rsqrtf(ss / 64.0f + eps);
    for (int i = lane; i < 64; i += 32) s[i] *= sc * (w ? w[i & 63] : 1.0f);
}

// per-head q/k norms (weighted) + plain norm on V. Block = position, warp = head.
static __global__ void k_headnorm(float *Q, float *K, float *V, const float *qn, const float *kn,
                                  int ne, float eps) {
    int p = blockIdx.x, hh = threadIdx.x >> 5, lane = threadIdx.x & 31;
    size_t at = (size_t)p * ne + hh * 64;
    d_rms64(Q + at, qn, eps, lane);
    d_rms64(K + at, kn, eps, lane);
    d_rms64(V + at, NULL, eps, lane);
}

// 2D NeoX RoPE: first 32 dims of each head rotate by patch column, second 32
// by row; NeoX pairs (i, i+16) inside each half. Block = position, warp = head.
static __global__ void k_rope2d(float *X, int ne, int n_cols, float theta) {
    int p = blockIdx.x, hh = threadIdx.x >> 5, r = threadIdx.x & 31;
    int sec = r >> 4, i = r & 15;
    int pos = sec ? p / n_cols : p % n_cols;
    float freq = __powf(theta, -2.0f * (float)i / 32.0f);
    float ang = (float)pos * freq, c = __cosf(ang), sn = __sinf(ang);
    float *s = X + (size_t)p * ne + hh * 64 + sec * 32;
    float a = s[i], b = s[i + 16];
    s[i]      = a * c - b * sn;
    s[i + 16] = a * sn + b * c;
}

// Full attention at scale 1.0, tensor-core flash. One CTA = 64 queries x one
// head; 4 warps, each owning 16 query rows END TO END — its own S tile, its
// own online max/sum, its own O accumulator, so there is no cross-warp
// reduction anywhere. K/V stream through shared in 32-key tiles as f16 (V
// staged TRANSPOSED so the PV b-fragments are packed pairs, like a W row); Q
// rides in registers as a-fragments for the whole key loop. Replaces a
// materialized-score kernel (one block per query x head, 2 FMAs per 5-shuffle
// dot) that cost more wall time than all seven GEMMs together — and with it
// goes its 48KB dynamic-shmem cap on the patch count.
static __global__ void k_attn(float *D, const float *Q, const float *K, const float *V,
                              int np, int ne) {
    __shared__ __half sk[32][72], svt[64][40];          // K tile; V tile, dim-major
    int hh = blockIdx.y, warp = threadIdx.x >> 5, g = (threadIdx.x & 31) >> 2, tig = threadIdx.x & 3;
    int qr = blockIdx.x * 64 + warp * 16;               // this warp's 16 query rows
    uint32_t qf[4][4];                                  // Q[16][64] as a-frags, 4 k-steps
    for (int ks = 0; ks < 4; ks++) {
        const float *q0 = Q + (size_t)(qr + g) * ne + hh * 64 + ks * 16 + tig * 2;
        const float *q8 = q0 + (size_t)8 * ne;
        qf[ks][0] = qr + g < np     ? vh_pkf(q0[0], q0[1]) : 0u;
        qf[ks][1] = qr + g + 8 < np ? vh_pkf(q8[0], q8[1]) : 0u;
        qf[ks][2] = qr + g < np     ? vh_pkf(q0[8], q0[9]) : 0u;
        qf[ks][3] = qr + g + 8 < np ? vh_pkf(q8[8], q8[9]) : 0u;
    }
    float acc[8][4] = { 0 };                            // O[16][64]: 8 n8 tiles
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.0f, l1 = 0.0f;   // rows g and g+8
    for (int kb = 0; kb < np; kb += 32) {
        for (int e = threadIdx.x; e < 32 * 16; e += 128) {   // stage K and V^T, f32 -> f16
            int r = e / 16, i = e % 16 * 4;                  // 4 dims per element
            const float *src = K + (size_t)(kb + r) * ne + hh * 64 + i;
            float4 kv = kb + r < np ? *(const float4 *)src : make_float4(0, 0, 0, 0);
            *(uint32_t *)&sk[r][i] = vh_pkf(kv.x, kv.y);
            *(uint32_t *)&sk[r][i + 2] = vh_pkf(kv.z, kv.w);
            src = V + (size_t)(kb + r) * ne + hh * 64 + i;
            float4 vv = kb + r < np ? *(const float4 *)src : make_float4(0, 0, 0, 0);
            svt[i][r] = __float2half(vv.x); svt[i + 1][r] = __float2half(vv.y);
            svt[i + 2][r] = __float2half(vv.z); svt[i + 3][r] = __float2half(vv.w);
        }
        __syncthreads();
        float s[4][4] = { 0 };                          // S[16][32]: 4 n8 tiles, f32
        for (int ks = 0; ks < 4; ks++)
            for (int kt = 0; kt < 4; kt++)
                vh_mma(s[kt],
                       qf[ks][0], qf[ks][1], qf[ks][2], qf[ks][3],
                       *(const uint32_t *)&sk[kt * 8 + g][ks * 16 + tig * 2],
                       *(const uint32_t *)&sk[kt * 8 + g][ks * 16 + tig * 2 + 8]);
        float hi0 = -1e30f, hi1 = -1e30f;               // online softmax, per query row
        for (int kt = 0; kt < 4; kt++) {                // mask the tail keys, find row maxes
            if (kb + kt * 8 + tig * 2 >= np)     s[kt][0] = s[kt][2] = -1e30f;
            if (kb + kt * 8 + tig * 2 + 1 >= np) s[kt][1] = s[kt][3] = -1e30f;
            hi0 = fmaxf(hi0, fmaxf(s[kt][0], s[kt][1]));
            hi1 = fmaxf(hi1, fmaxf(s[kt][2], s[kt][3]));
        }
        for (int o = 1; o < 4; o <<= 1) {               // max across the row's 4 lanes
            hi0 = fmaxf(hi0, __shfl_xor_sync(0xffffffffu, hi0, o));
            hi1 = fmaxf(hi1, __shfl_xor_sync(0xffffffffu, hi1, o));
        }
        float mn0 = fmaxf(m0, hi0), mn1 = fmaxf(m1, hi1);
        float al0 = __expf(m0 - mn0), al1 = __expf(m1 - mn1);
        m0 = mn0; m1 = mn1;
        float sum0 = 0.0f, sum1 = 0.0f;
        for (int kt = 0; kt < 4; kt++) {                // P = exp(S - m), rows rescaled
            s[kt][0] = __expf(s[kt][0] - m0); s[kt][1] = __expf(s[kt][1] - m0);
            s[kt][2] = __expf(s[kt][2] - m1); s[kt][3] = __expf(s[kt][3] - m1);
            sum0 += s[kt][0] + s[kt][1]; sum1 += s[kt][2] + s[kt][3];
        }
        for (int o = 1; o < 4; o <<= 1) {
            sum0 += __shfl_xor_sync(0xffffffffu, sum0, o);
            sum1 += __shfl_xor_sync(0xffffffffu, sum1, o);
        }
        l0 = l0 * al0 + sum0; l1 = l1 * al1 + sum1;
        for (int nt = 0; nt < 8; nt++)
            for (int c = 0; c < 4; c++) acc[nt][c] *= c < 2 ? al0 : al1;
        for (int ks = 0; ks < 2; ks++) {                // O += P.V, P refragmented in registers
            uint32_t p0 = vh_pkf(s[ks * 2][0], s[ks * 2][1]);
            uint32_t p1 = vh_pkf(s[ks * 2][2], s[ks * 2][3]);
            uint32_t p2 = vh_pkf(s[ks * 2 + 1][0], s[ks * 2 + 1][1]);
            uint32_t p3 = vh_pkf(s[ks * 2 + 1][2], s[ks * 2 + 1][3]);
            for (int nt = 0; nt < 8; nt++)
                vh_mma(acc[nt], p0, p1, p2, p3,
                       *(const uint32_t *)&svt[nt * 8 + g][ks * 16 + tig * 2],
                       *(const uint32_t *)&svt[nt * 8 + g][ks * 16 + tig * 2 + 8]);
        }
        __syncthreads();
    }
    float inv0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, inv1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    for (int nt = 0; nt < 8; nt++) {
        int i = hh * 64 + nt * 8 + tig * 2;
        if (qr + g < np) {
            D[(size_t)(qr + g) * ne + i] = acc[nt][0] * inv0;
            D[(size_t)(qr + g) * ne + i + 1] = acc[nt][1] * inv0;
        }
        if (qr + g + 8 < np) {
            D[(size_t)(qr + g + 8) * ne + i] = acc[nt][2] * inv1;
            D[(size_t)(qr + g + 8) * ne + i + 1] = acc[nt][3] * inv1;
        }
    }
}

// im2col from the raw RGB frame on device: the host used to build the f32
// patch matrix and upload it (7MB for a 624x480 legacy frame); now the 0.9MB
// frame uploads and this unpacks it. Layout matches the host encoders:
// channel-major within the patch, vec[(c*P+ky)*P+kx]. sc/ofs pick the range —
// (2,-1) for the legacy path's [-1,1], (1,0) for the unified path's [0,1] —
// applied in the host's exact op order (/255 first).
static __global__ void k_im2col(float *F, const uint8_t *rgb, int w, int n_cols, int P, int pin,
                                float sc, float ofs, size_t total) {
    for (size_t idx = blockIdx.x * 256ull + threadIdx.x; idx < total; idx += gridDim.x * 256ull) {
        int p = (int)(idx / pin), i = (int)(idx % pin);
        int col = p % n_cols, row = p / n_cols;
        int c = i / (P * P), ky = i / P % P, kx = i % P;
        F[idx] = (float)rgb[3 * ((size_t)(row * P + ky) * w + col * P + kx) + c] / 255.0f * sc + ofs;
    }
}

// LayerNorm one row: y = (x - mean) * rsqrt(var + eps) * w + b. Block per row.
static __global__ void k_lnorm(float *out, const float *in, const float *w, const float *b,
                               int n, float eps) {
    __shared__ float red[256];
    const float *x = in + (size_t)blockIdx.x * n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += 256) s += x[i];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int st = 128; st; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    float mean = red[0] / (float)n;
    __syncthreads();
    s = 0.0f;
    for (int i = threadIdx.x; i < n; i += 256) { float d = x[i] - mean; s += d * d; }
    red[threadIdx.x] = s;
    __syncthreads();
    for (int st = 128; st; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    float sc = rsqrtf(red[0] / (float)n + eps);
    float *y = out + (size_t)blockIdx.x * n;
    for (int i = threadIdx.x; i < n; i += 256) y[i] = (x[i] - mean) * sc * w[i] + b[i];
}

// X[row] += b, the same bias for every row.
static __global__ void k_addrow(float *X, const float *b, int ne, size_t total) {
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < total; i += gridDim.x * 256ull)
        X[i] += b[i % ne];
}

// i16 PCM -> f32 in [-1,1).
static __global__ void k_aframe(float *A, const int16_t *pcm, size_t total) {
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < total; i += gridDim.x * 256ull)
        A[i] = (float)pcm[i] / 32768.0f;
}

// X += learned position row: x-table[col] + y-table[row], straight from the
// once-uploaded position table (the host used to sum and upload these per frame).
static __global__ void k_posadd(float *X, const float *vpos, int n_cols, int pos_size, int ne, size_t total) {
    for (size_t idx = blockIdx.x * 256ull + threadIdx.x; idx < total; idx += gridDim.x * 256ull) {
        int p = (int)(idx / ne), i = (int)(idx % ne);
        X[idx] += vpos[(size_t)(p % n_cols) * ne + i] + vpos[((size_t)pos_size + p / n_cols) * ne + i];
    }
}

// 3x3 average pool over the patch grid, x sqrt(768). Block = output token.
static __global__ void k_pool(float *out, const float *X, int n_cols, int out_x, int ne, float sc) {
    int tx = blockIdx.x % out_x, ty = blockIdx.x / out_x;
    for (int i = threadIdx.x; i < ne; i += 256) {
        float s = 0.0f;
        for (int dy = 0; dy < 3; dy++)
            for (int dx = 0; dx < 3; dx++)
                s += X[(size_t)((ty * 3 + dy) * n_cols + (tx * 3 + dx)) * ne + i];
        out[(size_t)blockIdx.x * ne + i] = s * sc / 9.0f;
    }
}

// ---- upload / init ----------------------------------------------------------

static __half *up_h(const struct gguf_tensor *t) {     // weight -> device f16
    size_t n = 1;
    for (uint32_t i = 0; i < t->n_dims; i++) n *= t->dims[i];
    __half *d = NULL;
    if (cudaMalloc(&d, n * 2) != cudaSuccess) return NULL;
    if (t->type == GGML_TYPE_F16) {
        if (cudaMemcpy(d, t->data, n * 2, cudaMemcpyHostToDevice) != cudaSuccess) return NULL;
    } else {                                            // f32 (or any) -> f16 on host
        float *f = (float *)malloc(n * 4);
        __half *h = (__half *)malloc(n * 2);
        if (!f || !h || !dequantize_into(t->type, t->data, f, (int64_t)n)) { free(f); free(h); return NULL; }
        for (size_t i = 0; i < n; i++) h[i] = __float2half(f[i]);
        cudaError_t e = cudaMemcpy(d, h, n * 2, cudaMemcpyHostToDevice);
        free(f); free(h);
        if (e != cudaSuccess) return NULL;
    }
    return d;
}

static float *up_f(const float *src, size_t n) {       // norm weights -> device f32
    float *d = NULL;
    if (cudaMalloc(&d, n * 4) != cudaSuccess) return NULL;
    if (cudaMemcpy(d, src, n * 4, cudaMemcpyHostToDevice) != cudaSuccess) return NULL;
    return d;
}

static struct vcuda *vcuda_init(struct media *md) {
    if (md->v_embd / md->v_head != 64) {                // kernels assume dh = 64
        fprintf(stderr, "media-kernel: head dim %d != 64, using the host encoder\n", md->v_embd / md->v_head);
        return NULL;
    }
    struct vcuda *vc = (struct vcuda *)calloc(1, sizeof *vc);
    if (!vc) return NULL;
    vc->vl = (struct vld *)calloc((size_t)md->v_layer, sizeof *vc->vl);
    vc->patch16 = up_h(md->v_patch16);
    vc->mmv = up_h(md->mm_v);
    vc->vpos = up_f(md->v_pos, (size_t)2 * md->pos_size * md->v_embd);
    int ok = vc->vl && vc->patch16 && vc->mmv && vc->vpos;
    int ne = md->v_embd, dh = 64;
    for (int L = 0; ok && L < md->v_layer; L++) {
        const struct vlayer *s = &md->vl[L];
        struct vld *d = &vc->vl[L];
        ok = (d->ln1 = up_f(s->ln1, ne)) && (d->ln2 = up_f(s->ln2, ne)) &&
             (d->attn_post = up_f(s->attn_post, ne)) && (d->ffn_post = up_f(s->ffn_post, ne)) &&
             (d->qn = up_f(s->qn, dh)) && (d->kn = up_f(s->kn, dh)) &&
             (d->q = up_h(s->q)) && (d->k = up_h(s->k)) && (d->v = up_h(s->v)) &&
             (d->o = up_h(s->o)) && (d->gate = up_h(s->gate)) &&
             (d->up = up_h(s->up)) && (d->down = up_h(s->down)) != NULL;
    }
    if (!ok) {
        fprintf(stderr, "media-kernel: weight upload failed, using the host encoder\n");
        free(vc->vl); free(vc);                        // leaked partial uploads are acceptable here:
        return NULL;                                   // this only happens out-of-memory at startup
    }
    return vc;
}

static int ensure_bufs(struct vcuda *vc, int np, int ne, int n_embd) {
    if (np <= vc->np_cap) return 0;
    float **bufs[] = { &vc->F, &vc->X, &vc->H, &vc->Q, &vc->K, &vc->V, &vc->D,
                       &vc->G, &vc->U, &vc->pooled, &vc->rows };
    size_t sz[] = { (size_t)np * ne, (size_t)np * ne, (size_t)np * ne,
                    (size_t)np * ne, (size_t)np * ne, (size_t)np * ne, (size_t)np * ne,
                    (size_t)np * 4 * ne, (size_t)np * 4 * ne,
                    (size_t)(np / 9 + 1) * ne, (size_t)(np / 9 + 1) * n_embd };
    for (int i = 0; i < 11; i++) {
        cudaFree(*bufs[i]);
        if (cudaMalloc(bufs[i], sz[i] * 4) != cudaSuccess) { vc->np_cap = 0; return -1; }
    }
    cudaFree(vc->rgb);
    if (cudaMalloc(&vc->rgb, (size_t)np * ne) != cudaSuccess) { vc->np_cap = 0; return -1; }
    vc->np_cap = np;
    return 0;
}

// ---- the encoder ------------------------------------------------------------

extern "C" float *v_embed_image_gpu(struct media *md, const uint8_t *rgb, int w, int h, int *n_tokens) {
    if (md->gpu == (void *)-1) return NULL;            // init already failed once
    struct vcuda *vc = (struct vcuda *)md->gpu;
    if (!vc) {
        vc = vcuda_init(md);
        md->gpu = vc ? (void *)vc : (void *)-1;
        if (!vc) return NULL;
    }

    const int P = md->patch, ne = md->v_embd, nh = md->v_head, mg = md->v_merge;
    const float eps = 1e-6f;
    int n_cols = w / P, n_rows = h / P, np = n_cols * n_rows;
    if (n_cols > md->pos_size || n_rows > md->pos_size) {
        fprintf(stderr, "media: image %dx%d exceeds the %d-entry position table\n", w, h, md->pos_size);
        return NULL;
    }
    if (ensure_bufs(vc, np, ne, md->n_embd) != 0) return NULL;

    cudaMemcpy(vc->rgb, rgb, (size_t)np * ne, cudaMemcpyHostToDevice);   // ne = 3*P*P bytes/patch

    dim3 g_sq((ne + 63) / 64, (np + 63) / 64);          // [np][768] outputs
    dim3 g_up((4 * ne + 63) / 64, (np + 63) / 64);      // [np][3072]
    dim3 g_at((np + 63) / 64, nh);
    int nadd = (int)(((size_t)np * ne + 255) / 256);

    k_im2col<<<nadd, 256>>>(vc->F, vc->rgb, w, n_cols, P, ne, 2.0f, -1.0f, (size_t)np * ne);
    k_gemm<<<g_sq, 256>>>(vc->X, vc->F, vc->patch16, np, ne, ne);
    k_posadd<<<nadd, 256>>>(vc->X, vc->vpos, n_cols, md->pos_size, ne, (size_t)np * ne);

    for (int L = 0; L < md->v_layer; L++) {
        const struct vld *v = &vc->vl[L];
        k_rms<<<np, 256>>>(vc->H, vc->X, v->ln1, ne, eps);
        k_gemm<<<g_sq, 256>>>(vc->Q, vc->H, v->q, np, ne, ne);
        k_gemm<<<g_sq, 256>>>(vc->K, vc->H, v->k, np, ne, ne);
        k_gemm<<<g_sq, 256>>>(vc->V, vc->H, v->v, np, ne, ne);
        k_headnorm<<<np, nh * 32>>>(vc->Q, vc->K, vc->V, v->qn, v->kn, ne, eps);
        k_rope2d<<<np, nh * 32>>>(vc->Q, ne, n_cols, 100.0f);
        k_rope2d<<<np, nh * 32>>>(vc->K, ne, n_cols, 100.0f);
        k_attn<<<g_at, 128>>>(vc->D, vc->Q, vc->K, vc->V, np, ne);
        k_gemm<<<g_sq, 256>>>(vc->H, vc->D, v->o, np, ne, ne);
        k_rms<<<np, 256>>>(vc->H, vc->H, v->attn_post, ne, eps);
        k_add<<<nadd, 256>>>(vc->X, vc->H, (size_t)np * ne);
        k_rms<<<np, 256>>>(vc->H, vc->X, v->ln2, ne, eps);
        k_gemm<<<g_up, 256>>>(vc->G, vc->H, v->gate, np, ne, 4 * ne);
        k_gemm<<<g_up, 256>>>(vc->U, vc->H, v->up, np, ne, 4 * ne);
        k_gelu_mul<<<nadd * 4, 256>>>(vc->G, vc->U, (size_t)np * 4 * ne);
        k_gemm<<<g_sq, 256>>>(vc->D, vc->G, v->down, np, 4 * ne, ne);
        k_rms<<<np, 256>>>(vc->D, vc->D, v->ffn_post, ne, eps);
        k_add<<<nadd, 256>>>(vc->X, vc->D, (size_t)np * ne);
    }

    int out_x = n_cols / mg, out_y = n_rows / mg, n = out_x * out_y;
    k_pool<<<n, 256>>>(vc->pooled, vc->X, n_cols, out_x, ne, sqrtf((float)ne));
    k_gemm<<<dim3((md->n_embd + 63) / 64, (n + 63) / 64), 256>>>
        (vc->rows, vc->pooled, vc->mmv, n, ne, md->n_embd);
    k_rms<<<n, 256>>>(vc->rows, vc->rows, NULL, md->n_embd, eps);

    float *out = (float *)malloc((size_t)n * md->n_embd * 4);
    if (!out) return NULL;
    VC_CHECK(cudaMemcpy(out, vc->rows, (size_t)n * md->n_embd * 4, cudaMemcpyDeviceToHost));
    VC_CHECK(cudaGetLastError());
    *n_tokens = n;
    return out;
}

// ---- the unified path (gemma4uv / gemma4ua: the 12B) -------------------------
// No transformer at all: an image token is LayerNorm + linear of a raw 48x48
// patch, an audio token is rms + linear of a raw 40 ms frame — the 12B's own
// layers do the seeing (which is why its media spans need bidirectional
// attention). The host ground ~11 GFLOP per image through OpenMP matmats;
// here it is two k_gemm launches and change, and the audio clip's
// frame-at-a-time matvec loop becomes ONE batched GEMM.

struct uvcuda {
    __half *pw, *mmv, *aw;          // [pin -> ne] patch linear, [ne -> ne] projector, [frame -> ne] audio
    float *pb, *n1w, *n1b, *n2w, *n2b, *n3w, *n3b, *vpos;
    int np_cap, af_cap;             // work-buffer sizes, in patches / audio frames
    uint8_t *rgb;                   // [np * pin] the raw frame
    float *F, *E, *rows;            // [np][pin], [np][ne], [np][ne]
    int16_t *pcm;                   // [af * frame]
    float *A, *arows;               // [af][frame], [af][ne]
};

static struct uvcuda *uvcuda_init(struct media *md) {
    struct uvcuda *vc = (struct uvcuda *)calloc(1, sizeof *vc);
    if (!vc) return NULL;
    const int ne = md->n_embd, pin = md->patch * md->patch * 3;
    int ok = 1;
    if (md->v_patch_w)
        ok = (vc->pw = up_h(md->v_patch_w)) && (vc->mmv = up_h(md->mm_v)) &&
             (vc->pb = up_f(md->v_patch_b, ne)) &&
             (vc->n1w = up_f(md->vn1w, pin)) && (vc->n1b = up_f(md->vn1b, pin)) &&
             (vc->n2w = up_f(md->vn2w, ne)) && (vc->n2b = up_f(md->vn2b, ne)) &&
             (vc->n3w = up_f(md->vn3w, ne)) && (vc->n3b = up_f(md->vn3b, ne)) &&
             (vc->vpos = up_f(md->v_pos, (size_t)2 * md->pos_size * ne)) != NULL;
    if (ok && md->mm_a) ok = (vc->aw = up_h(md->mm_a)) != NULL;
    if (!ok) {
        fprintf(stderr, "media-kernel: weight upload failed, using the host encoder\n");
        free(vc);                                      // leaked partial uploads: OOM at startup only
        return NULL;
    }
    return vc;
}

// The lazy first-use init both unified embedders share (same latch as legacy).
static struct uvcuda *uv_state(struct media *md) {
    if (md->gpu == (void *)-1) return NULL;
    struct uvcuda *vc = (struct uvcuda *)md->gpu;
    if (!vc) {
        vc = uvcuda_init(md);
        md->gpu = vc ? (void *)vc : (void *)-1;
    }
    return vc;
}

extern "C" float *uv_embed_image_gpu(struct media *md, const uint8_t *rgb, int w, int h, int *n_tokens) {
    struct uvcuda *vc = uv_state(md);
    if (!vc || !vc->pw) return NULL;
    const int P = md->patch, pin = P * P * 3, ne = md->n_embd;
    int n_cols = w / P, n_rows = h / P, n = n_cols * n_rows;
    if (n_cols > md->pos_size || n_rows > md->pos_size) return NULL;   // host prints the error
    if (n > vc->np_cap) {
        cudaFree(vc->rgb); cudaFree(vc->F); cudaFree(vc->E); cudaFree(vc->rows);
        if (cudaMalloc(&vc->rgb, (size_t)n * pin) != cudaSuccess ||
            cudaMalloc(&vc->F, (size_t)n * pin * 4) != cudaSuccess ||
            cudaMalloc(&vc->E, (size_t)n * ne * 4) != cudaSuccess ||
            cudaMalloc(&vc->rows, (size_t)n * ne * 4) != cudaSuccess) { vc->np_cap = 0; return NULL; }
        vc->np_cap = n;
    }
    size_t tin = (size_t)n * pin, tne = (size_t)n * ne;
    cudaMemcpy(vc->rgb, rgb, tin, cudaMemcpyHostToDevice);
    dim3 gm((ne + 63) / 64, (n + 63) / 64);
    k_im2col<<<(int)((tin + 255) / 256), 256>>>(vc->F, vc->rgb, w, n_cols, P, pin, 1.0f, 0.0f, tin);
    k_lnorm<<<n, 256>>>(vc->F, vc->F, vc->n1w, vc->n1b, pin, 1e-5f);
    k_gemm<<<gm, 256>>>(vc->E, vc->F, vc->pw, n, pin, ne);
    k_addrow<<<(int)((tne + 255) / 256), 256>>>(vc->E, vc->pb, ne, tne);
    k_lnorm<<<n, 256>>>(vc->E, vc->E, vc->n2w, vc->n2b, ne, 1e-5f);
    k_posadd<<<(int)((tne + 255) / 256), 256>>>(vc->E, vc->vpos, n_cols, md->pos_size, ne, tne);
    k_lnorm<<<n, 256>>>(vc->E, vc->E, vc->n3w, vc->n3b, ne, 1e-5f);
    k_rms<<<n, 256>>>(vc->E, vc->E, NULL, ne, 1e-6f);
    k_gemm<<<gm, 256>>>(vc->rows, vc->E, vc->mmv, n, ne, ne);
    float *out = (float *)malloc(tne * 4);
    if (!out) return NULL;
    VC_CHECK(cudaMemcpy(out, vc->rows, tne * 4, cudaMemcpyDeviceToHost));
    VC_CHECK(cudaGetLastError());
    *n_tokens = n;
    return out;
}

extern "C" float *uv_embed_audio_gpu(struct media *md, const int16_t *pcm, int n_samples, int *n_tokens) {
    struct uvcuda *vc = uv_state(md);
    if (!vc || !vc->aw) return NULL;
    const int F = md->frame, ne = md->n_embd;
    int n = n_samples / F;
    if (n > vc->af_cap) {
        cudaFree(vc->pcm); cudaFree(vc->A); cudaFree(vc->arows);
        if (cudaMalloc(&vc->pcm, (size_t)n * F * 2) != cudaSuccess ||
            cudaMalloc(&vc->A, (size_t)n * F * 4) != cudaSuccess ||
            cudaMalloc(&vc->arows, (size_t)n * ne * 4) != cudaSuccess) { vc->af_cap = 0; return NULL; }
        vc->af_cap = n;
    }
    size_t tA = (size_t)n * F, tne = (size_t)n * ne;
    cudaMemcpy(vc->pcm, pcm, tA * 2, cudaMemcpyHostToDevice);
    k_aframe<<<(int)((tA + 255) / 256), 256>>>(vc->A, vc->pcm, tA);
    k_rms<<<n, 256>>>(vc->A, vc->A, NULL, F, 1e-6f);
    k_gemm<<<dim3((ne + 63) / 64, (n + 63) / 64), 256>>>(vc->arows, vc->A, vc->aw, n, F, ne);
    float *out = (float *)malloc(tne * 4);
    if (!out) return NULL;
    VC_CHECK(cudaMemcpy(out, vc->arows, tne * 4, cudaMemcpyDeviceToHost));
    VC_CHECK(cudaGetLastError());
    *n_tokens = n;
    return out;
}

static void a_gpu_free_state(struct media *md);   // gemma4a conformer state, below

extern "C" void v_gpu_free(struct media *md) {
    a_gpu_free_state(md);                          // audio state has its own pointer
    if (!md->gpu || md->gpu == (void *)-1) return;
    if (!md->legacy_v) {
        struct uvcuda *vc = (struct uvcuda *)md->gpu;
        cudaFree(vc->pw); cudaFree(vc->mmv); cudaFree(vc->aw);
        cudaFree(vc->pb); cudaFree(vc->n1w); cudaFree(vc->n1b); cudaFree(vc->n2w); cudaFree(vc->n2b);
        cudaFree(vc->n3w); cudaFree(vc->n3b); cudaFree(vc->vpos);
        cudaFree(vc->rgb); cudaFree(vc->F); cudaFree(vc->E); cudaFree(vc->rows);
        cudaFree(vc->pcm); cudaFree(vc->A); cudaFree(vc->arows);
        free(vc);
        md->gpu = NULL;
        return;
    }
    struct vcuda *vc = (struct vcuda *)md->gpu;
    for (int L = 0; L < md->v_layer; L++) {
        struct vld *d = &vc->vl[L];
        cudaFree(d->ln1); cudaFree(d->ln2); cudaFree(d->attn_post); cudaFree(d->ffn_post);
        cudaFree(d->qn); cudaFree(d->kn);
        cudaFree(d->q); cudaFree(d->k); cudaFree(d->v); cudaFree(d->o);
        cudaFree(d->gate); cudaFree(d->up); cudaFree(d->down);
    }
    cudaFree(vc->patch16); cudaFree(vc->mmv); cudaFree(vc->vpos);
    cudaFree(vc->rgb); cudaFree(vc->F); cudaFree(vc->X); cudaFree(vc->H);
    cudaFree(vc->Q); cudaFree(vc->K); cudaFree(vc->V); cudaFree(vc->D);
    cudaFree(vc->G); cudaFree(vc->U); cudaFree(vc->pooled); cudaFree(vc->rows);
    free(vc->vl);
    free(vc);
    md->gpu = NULL;
}

#include <cfloat>              // FLT_MAX: the inactive-clamp sentinel

// ===== gemma4a audio conformer (GPU) =========================================
// The 12-block conformer that media.c runs on the host (a_blocks_host), as CUDA
// kernels. The front end (log-mel + the two subsampling convs) stays on the
// host and hands us F[T][n_feat]; here we project in, run the stack, and
// project out to the LLM width. Rides the vision encoder's kernels where the
// shapes let it (k_gemm — which rounds activations to f16 for the tensor
// cores, the same relaxed-numerics class as the vision path — plus k_rms,
// k_add, k_addrow); the host a_blocks_host stays in as the LG_MEDIA_VERIFY
// oracle. Ported back from the audio-encoder branch (4418a6e) when the QAT
// releases fixed the encoder; only reachable under LG_GEMMA4A=1 (media.c).

struct ald {                                  // device mirror of struct alayer
    float *ffn_norm, *ffn_post, *ffn_norm1, *ffn_post1;
    float *attn_pre, *attn_post, *pds;        // pds [dh]
    float *norm_conv, *conv_norm, *ln2, *dw;  // dw [ne][5]
    __half *ffn_up, *ffn_down, *ffn_up1, *ffn_down1;
    __half *q, *k, *v, *o, *k_rel, *pw1, *pw2;
};

struct acuda {
    struct ald *al;
    __half *inp_proj, *out_proj, *mm_a;
    float *out_proj_b, *rpe;                  // out_proj_b [ao], rpe [13][ne]
    int ne, nh, dh, n_layer, n_feat, ao, n_embd;
    int t_cap;                                // size of the work buffers, in frames
    float *F, *X, *H, *G, *D, *Q, *K, *V, *P, *Y, *rows;
    float *S;                                 // clamp scratch (q/k/v share their input)
};

// dst = clamp(src, lo, hi); dst == src is fine (same-index read/write).
static __global__ void k_clamp(float *dst, const float *src, float lo, float hi, size_t total) {
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < total; i += gridDim.x * 256ull) {
        float v = src[i];
        dst[i] = v < lo ? lo : v > hi ? hi : v;
    }
}

// The QAT clippable-linear halves: input clamps src into tmp (or in place),
// output clamps in place. Inactive ranges (+/-FLT_MAX, pre-QAT files) launch
// nothing — the gemm reads the original buffer.
static const float *cl_in(float *tmp, const float *x, size_t n, const struct aclamp *c) {
    if (c->ilo <= -FLT_MAX && c->ihi >= FLT_MAX) return x;
    k_clamp<<<(int)((n + 255) / 256), 256>>>(tmp, x, c->ilo, c->ihi, n);
    return tmp;
}
static void cl_out(float *x, size_t n, const struct aclamp *c) {
    if (c->olo <= -FLT_MAX && c->ohi >= FLT_MAX) return;
    k_clamp<<<(int)((n + 255) / 256), 256>>>(x, x, c->olo, c->ohi, n);
}

static __global__ void k_silu(float *x, size_t total) {
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < total; i += gridDim.x * 256ull) {
        float v = x[i]; x[i] = v / (1.0f + __expf(-v));
    }
}

// x += s * a   (s = 0.5 for the macaron half-step FFNs, 1.0 for full residuals).
static __global__ void k_madd(float *x, const float *a, float s, size_t total) {
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < total; i += gridDim.x * 256ull) x[i] += s * a[i];
}

// Q *= q_scale * per_dim_scale[i % dh];  K *= k_scale.  P (the rel-pos term) is
// left unscaled, exactly as the host does — the score is q_scaled . (k_scaled + p).
static __global__ void k_qk_scale(float *Q, float *K, const float *pds, int dh,
                                  float qs, float ks, size_t total) {
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < total; i += gridDim.x * 256ull) {
        Q[i] *= qs * pds[i % dh];
        K[i] *= ks;
    }
}

// GLU: D[t][i] = G[t][i] * sigmoid(G[t][ne + i])   (G is [T][2 ne]).
static __global__ void k_glu(float *D, const float *G, int ne, size_t total) {
    for (size_t idx = blockIdx.x * 256ull + threadIdx.x; idx < total; idx += gridDim.x * 256ull) {
        int i = idx % ne; size_t t = idx / ne;
        const float *g = G + t * 2 * ne;
        D[idx] = g[i] / (1.0f + __expf(-g[ne + i]));
    }
}

// Causal depthwise conv, 5 taps: H[t][i] = sum_j D[t-4+j][i] * dw[i][j], ts >= 0.
static __global__ void k_depthwise(float *H, const float *D, const float *dw, int ne, size_t total) {
    for (size_t idx = blockIdx.x * 256ull + threadIdx.x; idx < total; idx += gridDim.x * 256ull) {
        int i = idx % ne, t = (int)(idx / ne);
        float s = 0.0f;
        for (int j = 0; j < 5; j++) {
            int ts = t - 4 + j;
            if (ts >= 0) s += D[(size_t)ts * ne + i] * dw[(size_t)i * 5 + j];
        }
        H[idx] = s;
    }
}

// Chunked local attention: each frame t attends to the causal 12-frame window
// [t-11, t]. Block = (frame, head), blockDim.x = dh (a power of 2, <= 1024).
// score = 50*tanh(q.(k+p)/50) over the window, softmax, then the V-weighted sum;
// p is the projected rel-pos row for distance t-gk (table index 12-(t-gk), 12 = self).
static __global__ void k_local_attn(float *Dout, const float *Q, const float *K, const float *V,
                                    const float *P, int ne, int dh) {
    int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    __shared__ float red[1024];
    __shared__ float att[12];
    __shared__ float ssum;
    const float *qh = Q + (size_t)t * ne + h * dh;
    float qv = qh[i];
    int k0 = t - 11 > 0 ? t - 11 : 0, nk = t - k0 + 1;
    for (int g = 0; g < nk; g++) {                     // 12 windowed dot products
        int gk = k0 + g;
        const float *kh = K + (size_t)gk * ne + h * dh;
        const float *ph = P + (size_t)(12 - (t - gk)) * ne + h * dh;
        red[i] = qv * (kh[i] + ph[i]);
        __syncthreads();
        for (int s = dh >> 1; s; s >>= 1) { if (i < s) red[i] += red[i + s]; __syncthreads(); }
        if (i == 0) att[g] = 50.0f * tanhf(red[0] / 50.0f);
        __syncthreads();
    }
    if (i == 0) {                                      // softmax over the <=12 scores
        float mx = -1e30f;
        for (int g = 0; g < nk; g++) if (att[g] > mx) mx = att[g];
        float sum = 0.0f;
        for (int g = 0; g < nk; g++) { att[g] = expf(att[g] - mx); sum += att[g]; }
        ssum = sum;
    }
    __syncthreads();
    float o = 0.0f;
    for (int g = 0; g < nk; g++) o += att[g] * V[(size_t)(k0 + g) * ne + h * dh + i];
    Dout[(size_t)t * ne + h * dh + i] = o / ssum;
}

// ---- audio upload / init ----------------------------------------------------

static struct acuda *acuda_init(struct media *md) {
    int ne = md->a_embd, nh = md->a_head, dh = ne / nh;
    int ao = (int)md->a_out_proj->dims[1];
    if (dh > 1024 || (dh & (dh - 1)) != 0) {       // the local-attn reduction wants a pow2 dh
        fprintf(stderr, "media-cuda: audio head dim %d unsupported, using the host conformer\n", dh);
        return NULL;
    }
    if ((ne | ao | md->n_embd) & 1) {              // k_gemm stages/writes in pairs: dims must be even
        fprintf(stderr, "media-cuda: odd audio width (%d/%d/%d), using the host conformer\n",
                ne, ao, md->n_embd);
        return NULL;
    }
    struct acuda *ac = (struct acuda *)calloc(1, sizeof *ac);
    if (!ac) return NULL;
    ac->al = (struct ald *)calloc((size_t)md->a_layer, sizeof *ac->al);
    ac->ne = ne; ac->nh = nh; ac->dh = dh; ac->n_layer = md->a_layer;
    ac->ao = ao;
    ac->n_embd = md->n_embd;
    ac->inp_proj = up_h(md->a_inp_proj);
    ac->out_proj = up_h(md->a_out_proj);
    ac->mm_a     = up_h(md->mm_a);
    ac->out_proj_b = up_f(md->a_out_proj_b, (size_t)ac->ao);
    ac->rpe        = up_f(md->rpe, (size_t)13 * ne);
    int ok = ac->al && ac->inp_proj && ac->out_proj && ac->mm_a && ac->out_proj_b && ac->rpe;
    for (int L = 0; ok && L < md->a_layer; L++) {
        const struct alayer *s = &md->al[L];
        struct ald *d = &ac->al[L];
        ok = (d->ffn_norm = up_f(s->ffn_norm, ne)) && (d->ffn_post = up_f(s->ffn_post, ne)) &&
             (d->ffn_norm1 = up_f(s->ffn_norm1, ne)) && (d->ffn_post1 = up_f(s->ffn_post1, ne)) &&
             (d->attn_pre = up_f(s->attn_pre, ne)) && (d->attn_post = up_f(s->attn_post, ne)) &&
             (d->pds = up_f(s->pds, dh)) && (d->norm_conv = up_f(s->norm_conv, ne)) &&
             (d->conv_norm = up_f(s->conv_norm, ne)) && (d->ln2 = up_f(s->ln2, ne)) &&
             (d->dw = up_f(s->dw, (size_t)ne * 5)) &&
             (d->ffn_up = up_h(s->ffn_up)) && (d->ffn_down = up_h(s->ffn_down)) &&
             (d->ffn_up1 = up_h(s->ffn_up1)) && (d->ffn_down1 = up_h(s->ffn_down1)) &&
             (d->q = up_h(s->q)) && (d->k = up_h(s->k)) && (d->v = up_h(s->v)) &&
             (d->o = up_h(s->o)) && (d->k_rel = up_h(s->k_rel)) &&
             (d->pw1 = up_h(s->pw1)) && (d->pw2 = up_h(s->pw2)) != NULL;
    }
    if (!ok) {
        fprintf(stderr, "media-cuda: audio weight upload failed, using the host conformer\n");
        free(ac->al); free(ac);                    // partial uploads leaked (startup OOM only)
        return NULL;
    }
    return ac;
}

static int ensure_abufs(struct acuda *ac, int T) {
    if (T <= ac->t_cap) return 0;
    int ne = ac->ne, ao = ac->ao, nf = ac->n_feat, ce = ac->n_embd;
    float **bufs[] = { &ac->F, &ac->X, &ac->H, &ac->G, &ac->D, &ac->Q, &ac->K, &ac->V,
                       &ac->P, &ac->Y, &ac->rows, &ac->S };
    size_t sz[] = { (size_t)T * nf, (size_t)T * ne, (size_t)T * ne, (size_t)T * 4 * ne,
                    (size_t)T * 2 * ne, (size_t)T * ne, (size_t)T * ne, (size_t)T * ne,
                    (size_t)13 * ne, (size_t)T * ao, (size_t)T * ce, (size_t)T * ne };
    for (int i = 0; i < 12; i++) {
        cudaFree(*bufs[i]);
        if (cudaMalloc(bufs[i], sz[i] * 4) != cudaSuccess) { ac->t_cap = 0; return -1; }
    }
    ac->t_cap = T;
    return 0;
}

// ---- the conformer ----------------------------------------------------------

extern "C" float *a_blocks_gpu(struct media *md, const float *F, int T, int n_feat, int *n_tokens) {
    if (md->audio_gpu == (void *)-1) return NULL;      // init already failed once
    struct acuda *ac = (struct acuda *)md->audio_gpu;
    if (!ac) {
        ac = acuda_init(md);
        md->audio_gpu = ac ? (void *)ac : (void *)-1;
        if (!ac) return NULL;
    }
    const int ne = ac->ne, nh = ac->nh, dh = ac->dh, ao = ac->ao, n_embd = ac->n_embd;
    const float eps = 1e-6f;
    if (n_feat & 1) return NULL;                       // k_gemm stages K in pairs
    ac->n_feat = n_feat;
    if (ensure_abufs(ac, T) != 0) return NULL;
    cudaMemcpy(ac->F, F, (size_t)T * n_feat * 4, cudaMemcpyHostToDevice);

    #define AGG(rows, cols) dim3(((cols) + 63) / 64, ((rows) + 63) / 64)
    const size_t Tne = (size_t)T * ne;
    const int e_ne  = (int)((Tne + 255) / 256);                    // elementwise grid, [T][ne]
    const int e_4ne = (int)(((size_t)T * 4 * ne + 255) / 256);
    const float q_scale = (1.0f / sqrtf((float)dh)) / logf(2.0f);  // same constants as the host
    const float k_scale = logf(1.0f + expf(1.0f)) / logf(2.0f);

    k_gemm<<<AGG(T, ne), 256>>>(ac->X, ac->F, ac->inp_proj, T, n_feat, ne);   // project F in

    for (int L = 0; L < ac->n_layer; L++) {
        const struct ald *a = &ac->al[L];
        const struct alayer *cl = &md->al[L];              // the QAT clamp quads

        // FFN 1 (half-step residual)
        k_rms<<<T, 256>>>(ac->H, ac->X, a->ffn_norm, ne, eps);
        k_gemm<<<AGG(T, 4 * ne), 256>>>(ac->G, cl_in(ac->H, ac->H, Tne, &cl->c_ffn_up), a->ffn_up, T, ne, 4 * ne);
        cl_out(ac->G, (size_t)T * 4 * ne, &cl->c_ffn_up);
        k_silu<<<e_4ne, 256>>>(ac->G, (size_t)T * 4 * ne);
        k_gemm<<<AGG(T, ne), 256>>>(ac->D, cl_in(ac->G, ac->G, (size_t)T * 4 * ne, &cl->c_ffn_down), a->ffn_down, T, 4 * ne, ne);
        cl_out(ac->D, Tne, &cl->c_ffn_down);
        k_rms<<<T, 256>>>(ac->D, ac->D, a->ffn_post, ne, eps);
        k_madd<<<e_ne, 256>>>(ac->X, ac->D, 0.5f, Tne);

        // chunked local attention with relative positions
        k_rms<<<T, 256>>>(ac->H, ac->X, a->attn_pre, ne, eps);
        k_gemm<<<AGG(T, ne), 256>>>(ac->Q, cl_in(ac->S, ac->H, Tne, &cl->c_q), a->q, T, ne, ne);
        cl_out(ac->Q, Tne, &cl->c_q);
        k_gemm<<<AGG(T, ne), 256>>>(ac->K, cl_in(ac->S, ac->H, Tne, &cl->c_k), a->k, T, ne, ne);
        cl_out(ac->K, Tne, &cl->c_k);
        k_gemm<<<AGG(T, ne), 256>>>(ac->V, cl_in(ac->S, ac->H, Tne, &cl->c_v), a->v, T, ne, ne);
        cl_out(ac->V, Tne, &cl->c_v);
        k_gemm<<<AGG(13, ne), 256>>>(ac->P, ac->rpe, a->k_rel, 13, ne, ne);   // k_rel: no QAT ranges
        k_qk_scale<<<e_ne, 256>>>(ac->Q, ac->K, a->pds, dh, q_scale, k_scale, Tne);
        k_local_attn<<<dim3(T, nh), dh>>>(ac->D, ac->Q, ac->K, ac->V, ac->P, ne, dh);
        k_gemm<<<AGG(T, ne), 256>>>(ac->H, cl_in(ac->D, ac->D, Tne, &cl->c_o), a->o, T, ne, ne);
        cl_out(ac->H, Tne, &cl->c_o);
        k_rms<<<T, 256>>>(ac->H, ac->H, a->attn_post, ne, eps);
        k_add<<<e_ne, 256>>>(ac->X, ac->H, Tne);

        // convolution module
        k_rms<<<T, 256>>>(ac->H, ac->X, a->norm_conv, ne, eps);
        k_gemm<<<AGG(T, 2 * ne), 256>>>(ac->G, cl_in(ac->H, ac->H, Tne, &cl->c_pw1), a->pw1, T, ne, 2 * ne);
        cl_out(ac->G, (size_t)T * 2 * ne, &cl->c_pw1);
        k_glu<<<e_ne, 256>>>(ac->D, ac->G, ne, Tne);
        k_depthwise<<<e_ne, 256>>>(ac->H, ac->D, a->dw, ne, Tne);
        k_rms<<<T, 256>>>(ac->H, ac->H, a->conv_norm, ne, eps);
        k_silu<<<e_ne, 256>>>(ac->H, Tne);
        k_gemm<<<AGG(T, ne), 256>>>(ac->D, cl_in(ac->H, ac->H, Tne, &cl->c_pw2), a->pw2, T, ne, ne);
        cl_out(ac->D, Tne, &cl->c_pw2);
        k_add<<<e_ne, 256>>>(ac->X, ac->D, Tne);

        // FFN 2 (half-step residual)
        k_rms<<<T, 256>>>(ac->H, ac->X, a->ffn_norm1, ne, eps);
        k_gemm<<<AGG(T, 4 * ne), 256>>>(ac->G, cl_in(ac->H, ac->H, Tne, &cl->c_ffn_up1), a->ffn_up1, T, ne, 4 * ne);
        cl_out(ac->G, (size_t)T * 4 * ne, &cl->c_ffn_up1);
        k_silu<<<e_4ne, 256>>>(ac->G, (size_t)T * 4 * ne);
        k_gemm<<<AGG(T, ne), 256>>>(ac->D, cl_in(ac->G, ac->G, (size_t)T * 4 * ne, &cl->c_ffn_down1), a->ffn_down1, T, 4 * ne, ne);
        cl_out(ac->D, Tne, &cl->c_ffn_down1);
        k_rms<<<T, 256>>>(ac->D, ac->D, a->ffn_post1, ne, eps);
        k_madd<<<e_ne, 256>>>(ac->X, ac->D, 0.5f, Tne);

        // layer output norm
        k_rms<<<T, 256>>>(ac->X, ac->X, a->ln2, ne, eps);
    }

    // out proj -> + bias -> plain RMS -> the LLM-width projection
    k_gemm<<<AGG(T, ao), 256>>>(ac->Y, ac->X, ac->out_proj, T, ne, ao);
    k_addrow<<<(int)(((size_t)T * ao + 255) / 256), 256>>>(ac->Y, ac->out_proj_b, ao, (size_t)T * ao);
    k_rms<<<T, 256>>>(ac->Y, ac->Y, NULL, ao, eps);
    k_gemm<<<AGG(T, n_embd), 256>>>(ac->rows, ac->Y, ac->mm_a, T, ao, n_embd);
    #undef AGG

    float *out = (float *)malloc((size_t)T * n_embd * 4);
    if (!out) return NULL;
    VC_CHECK(cudaMemcpy(out, ac->rows, (size_t)T * n_embd * 4, cudaMemcpyDeviceToHost));
    VC_CHECK(cudaGetLastError());
    *n_tokens = T;
    return out;
}

static void a_gpu_free_state(struct media *md) {
    struct acuda *ac = (struct acuda *)md->audio_gpu;
    if (!ac || md->audio_gpu == (void *)-1) { md->audio_gpu = NULL; return; }
    for (int L = 0; L < ac->n_layer; L++) {
        struct ald *d = &ac->al[L];
        cudaFree(d->ffn_norm); cudaFree(d->ffn_post); cudaFree(d->ffn_norm1); cudaFree(d->ffn_post1);
        cudaFree(d->attn_pre); cudaFree(d->attn_post); cudaFree(d->pds);
        cudaFree(d->norm_conv); cudaFree(d->conv_norm); cudaFree(d->ln2); cudaFree(d->dw);
        cudaFree(d->ffn_up); cudaFree(d->ffn_down); cudaFree(d->ffn_up1); cudaFree(d->ffn_down1);
        cudaFree(d->q); cudaFree(d->k); cudaFree(d->v); cudaFree(d->o); cudaFree(d->k_rel);
        cudaFree(d->pw1); cudaFree(d->pw2);
    }
    cudaFree(ac->inp_proj); cudaFree(ac->out_proj); cudaFree(ac->mm_a);
    cudaFree(ac->out_proj_b); cudaFree(ac->rpe);
    cudaFree(ac->F); cudaFree(ac->X); cudaFree(ac->H); cudaFree(ac->G); cudaFree(ac->D);
    cudaFree(ac->Q); cudaFree(ac->K); cudaFree(ac->V); cudaFree(ac->P); cudaFree(ac->Y);
    cudaFree(ac->rows); cudaFree(ac->S);
    free(ac->al);
    free(ac);
    md->audio_gpu = NULL;
}
