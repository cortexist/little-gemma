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

extern "C" {
#include "model.h"
#include "quant.h"
}

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

struct actq { int8_t *xq; float *xd; int *xs; };
static const struct actq AQ0 = { NULL, NULL, NULL };

// Quantize one 32-element group g (global index) of x into int8 + scale + sum.
__device__ static void d_quant_group(const float *xb, int g, struct actq aq) {
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(xb[i]));
    float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
    int8_t *q = aq.xq + (size_t)g * 32;
    int sum = 0;
    for (int i = 0; i < 32; i++) { int v = __float2int_rn(xb[i] * id); q[i] = (int8_t)v; sum += v; }
    aq.xd[g] = d; aq.xs[g] = sum;
}

// ====================  non-matmul compute kernels  =========================

// RMSNorm over `rows` vectors of length n. w (length n, shared across rows) or NULL.
// One block (256 threads) per row.
__global__ static void rmsnorm_kernel(float *out, const float *x, const float *w, int n, float eps, struct actq aq) {
    int row = blockIdx.x;
    const float *xr = x + (size_t)row * n;
    float *outr = out + (size_t)row * n;
    __shared__ float sh[256];
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
__global__ static void rmsnorm_add_kernel(float *acc, const float *x, const float *w, int n, float eps,
                                          const float *os, struct actq aq) {
    __shared__ float sh[256];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    sh[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s]; __syncthreads(); }
    float scale = rsqrtf(sh[0] / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        // __fadd_rn keeps the separately-rounded mul-then-add of the unfused
        // kernels (an FMA here would change the last bit, and the greedy path).
        float v = __fadd_rn(acc[i], x[i] * scale * w[i]);
        acc[i] = os ? v * os[0] : v;
    }
    if (aq.xq) {
        __syncthreads();
        for (int g = threadIdx.x; g < n / 32; g += blockDim.x) d_quant_group(acc + g * 32, g, aq);
    }
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

// Attention for the current query position: one block per query head. pos is device-
// resident and start is derived in-kernel (so the launch params never change across
// tokens -> the forward captures into a static CUDA graph). window=0 means full causal.
// Shared memory holds the (pos-start+1) scores; the launch sizes it for max_seq. Scale 1.0.
__global__ static void attn_kernel(float *xb, const float *q, const float *Kc, const float *Vc,
                                   int hd, int kv_dim, int gqa, const int *d_pos, int window, struct actq aq) {
    int hh = blockIdx.x, kvh = hh / gqa;
    const float *qh = q + (size_t)hh * hd;
    int pos = *d_pos;
    int start = (window > 0 && pos - window + 1 > 0) ? pos - window + 1 : 0;
    int T = pos - start + 1;
    extern __shared__ float att[];
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        const float *kt = Kc + (size_t)(start + t) * kv_dim + (size_t)kvh * hd;
        float s = 0.0f; for (int i = 0; i < hd; i++) s += qh[i] * kt[i];
        att[t] = s;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float mx = att[0]; for (int t = 1; t < T; t++) if (att[t] > mx) mx = att[t];
        float sum = 0.0f; for (int t = 0; t < T; t++) { att[t] = expf(att[t] - mx); sum += att[t]; }
        float inv = 1.0f / sum; for (int t = 0; t < T; t++) att[t] *= inv;
    }
    __syncthreads();
    float *outh = xb + (size_t)hh * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < T; t++) { const float *vt = Vc + (size_t)(start + t) * kv_dim + (size_t)kvh * hd; acc += att[t] * vt[i]; }
        outh[i] = acc;
    }
    if (aq.xq) {
        __syncthreads();
        for (int g = threadIdx.x; g < hd / 32; g += blockDim.x)
            d_quant_group(outh + g * 32, (hh * hd) / 32 + g, aq);
    }
}

// Write one row (length n) into the kv cache at the device-resident position. Replaces
// a pos-offset cudaMemcpy so the node is static (constant args) and graph-capturable.
__global__ static void kv_write_kernel(float *dst, const float *src, const int *d_pos, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[(size_t)(*d_pos) * n + i] = src[i];
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
__global__ static void add_kernel(float *a, const float *b, int n, struct actq aq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
    d_quant_block(a, n, aq);
}
__global__ static void geglu_kernel(float *g, const float *u, int n, struct actq aq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = d_gelu(g[i]) * u[i];
    d_quant_block(g, n, aq);
}
__global__ static void scale_const_kernel(float *a, float s, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) a[i] *= s; }
__global__ static void combine_kernel(float *out, const float *p, const float *t, float c, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = (p[i] + t[i]) * c; }
__global__ static void softcap_kernel(float *l, float sc, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) l[i] = sc * tanhf(l[i] / sc); }

// ====================  device state: weights + scratch  =====================

static const struct gguf_context *g_ctx = NULL;
static unsigned char *d_blob = NULL;

static void ensure_weights(struct model *m) {
    if (d_blob) return;
    g_ctx = m->ctx;
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

// Resident device activation scratch (allocated once, reused across tokens).
static float *dx, *dh, *dq, *dkb, *dvb, *dxb, *dout, *dg1, *dg2, *dpg, *dlogits;
static float *d_ipl, *d_tok, *d_proj;  // per-layer-input (PLE) buffers
static int *d_pos;                      // device-resident token position (for static graph)
static int scratch_ok = 0;

static void ensure_scratch(struct model *m) {
    if (scratch_ok) return;
    const struct config *c = &m->cfg;
    int maxhd = 0, maxkv = 0;
    for (int L = 0; L < c->n_layer; L++) {
        if (m->head_dim[L] > maxhd) maxhd = m->head_dim[L];
        int kvd = m->n_head_kv[L] * m->head_dim[L];
        if (kvd > maxkv) maxkv = kvd;
    }
    int q_max = c->n_head * maxhd, ne = c->n_embd, nff = c->n_ff, ple = c->n_embd_per_layer;
    CUDA_CHECK(cudaMalloc(&dx,  (size_t)ne   * 4)); CUDA_CHECK(cudaMalloc(&dh,  (size_t)ne   * 4));
    CUDA_CHECK(cudaMalloc(&dout,(size_t)ne   * 4)); CUDA_CHECK(cudaMalloc(&dq,  (size_t)q_max * 4));
    CUDA_CHECK(cudaMalloc(&dxb, (size_t)q_max * 4)); CUDA_CHECK(cudaMalloc(&dkb, (size_t)maxkv * 4));
    CUDA_CHECK(cudaMalloc(&dvb, (size_t)maxkv * 4)); CUDA_CHECK(cudaMalloc(&dg1, (size_t)nff * 4));
    CUDA_CHECK(cudaMalloc(&dg2, (size_t)nff * 4));   CUDA_CHECK(cudaMalloc(&dlogits, (size_t)c->n_vocab * 4));
    CUDA_CHECK(cudaMalloc(&d_pos, sizeof(int)));
    if (ple > 0) {
        size_t total = (size_t)ple * c->n_layer;
        CUDA_CHECK(cudaMalloc(&dpg,   (size_t)ple * 4));
        CUDA_CHECK(cudaMalloc(&d_ipl, total * 4));
        CUDA_CHECK(cudaMalloc(&d_tok, total * 4));
        CUDA_CHECK(cudaMalloc(&d_proj,total * 4));
    }
    scratch_ok = 1;
}

// The backend seam — the including .cu defines these three.
// matmul_q: d_out[m] = W . d_x with W the quantized tensor t (all device ptrs).
//   The int8 backend reads d_x's quantization from its buffers, already filled
//   by a producer kernel's epilogue (actq_for) or by act_quantize.
// actq_for(k): the backend's epilogue buffers for a k-length activation that is
//   about to feed matmul_q (the f32 backend returns AQ0 — no quantization).
// act_quantize: quantize d_x explicitly — for the one activation no kernel
//   produces (the host-uploaded embedding); no-op for f32.
static void matmul_q(float *d_out, const struct gguf_tensor *t, const float *d_x, int k, int m);
static struct actq actq_for(int k);
static void act_quantize(const float *d_x, int k);

// ====================  kv cache (device buffers)  ===========================

extern "C" int kvcache_init(struct kvcache *kv, const struct model *m, int max_seq) {
    const struct config *c = &m->cfg;
    kv->n_layer = c->n_layer; kv->max_seq = max_seq;
    kv->kv_dim = (int *)calloc((size_t)c->n_layer, sizeof(int));
    kv->k = (float **)calloc((size_t)c->n_layer, sizeof(float *));  // host array of device ptrs
    kv->v = (float **)calloc((size_t)c->n_layer, sizeof(float *));
    if (!kv->kv_dim || !kv->k || !kv->v) return -1;
    for (int L = 0; L < c->n_layer; L++) {
        kv->kv_dim[L] = m->n_head_kv[L] * m->head_dim[L];
        size_t bytes = (size_t)max_seq * kv->kv_dim[L] * sizeof(float);
        CUDA_CHECK(cudaMalloc(&kv->k[L], bytes)); CUDA_CHECK(cudaMemset(kv->k[L], 0, bytes));
        CUDA_CHECK(cudaMalloc(&kv->v[L], bytes)); CUDA_CHECK(cudaMemset(kv->v[L], 0, bytes));
    }
    return 0;
}
extern "C" void kvcache_free(struct kvcache *kv) {
    if (!kv) return;
    for (int L = 0; L < kv->n_layer; L++) { cudaFree(kv->k[L]); cudaFree(kv->v[L]); }
    free(kv->k); free(kv->v); free(kv->kv_dim);
    kv->k = kv->v = NULL; kv->kv_dim = NULL;
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

// Device-only slice of the forward — the part a CUDA graph will capture: all layers
// plus the final norm and output projection. dx already holds the scaled embedding and
// d_ipl the PLE inputs; every op runs on the (per-thread) default stream with no host
// work or sync, so it is stream-capturable. KV-cache writes use async copies for the
// same reason (a synchronous cudaMemcpy is illegal mid-capture).
static void forward_layers_and_head(struct model *m, struct kvcache *kv) {
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
        rmsnorm_kernel<<<1, 256>>>(dh, dx, dW_layer(m, L, "attn_norm.weight"), n_embd, eps, actq_for(n_embd));
        matmul_q(dq, wq_layer(m, L, "attn_q.weight"), dh, n_embd, q_dim);
        rmsnorm_kernel<<<n_head, 256>>>(dq, dq, dW_layer(m, L, "attn_q_norm.weight"), hd, eps, AQ0);
        rope_kernel<<<gridn(n_head * hd / 2), 256>>>(dq, hd / 2, hd, d_pos, base, ff, n_head * hd / 2);

        int src = kv_src_dev(m, L);
        if (L < c->n_kv_start) {
            matmul_q(dkb, wq_layer(m, L, "attn_k.weight"), dh, n_embd, kv_dim);   // dh still quantized
            const struct gguf_tensor *wv = wq_layer(m, L, "attn_v.weight");
            if (wv) matmul_q(dvb, wv, dh, n_embd, kv_dim);
            else    CUDA_CHECK(cudaMemcpyAsync(dvb, dkb, (size_t)kv_dim * 4, cudaMemcpyDeviceToDevice, 0));
            rmsnorm_kernel<<<n_head_kv, 256>>>(dkb, dkb, dW_layer(m, L, "attn_k_norm.weight"), hd, eps, AQ0);
            rmsnorm_kernel<<<n_head_kv, 256>>>(dvb, dvb, NULL, hd, eps, AQ0);  // plain V norm
            rope_kernel<<<gridn(n_head_kv * hd / 2), 256>>>(dkb, hd / 2, hd, d_pos, base, ff, n_head_kv * hd / 2);
            kv_write_kernel<<<gridn(kv_dim), 256>>>(kv->k[L], dkb, d_pos, kv_dim);    // dst offset from d_pos
            kv_write_kernel<<<gridn(kv_dim), 256>>>(kv->v[L], dvb, d_pos, kv_dim);
        }
        const float *Kc = kv->k[src], *Vc = kv->v[src];
        int gqa = n_head / n_head_kv;
        int window = (local && c->sliding_window > 0) ? c->sliding_window : 0;
        attn_kernel<<<n_head, 128, (size_t)kv->max_seq * sizeof(float)>>>(dxb, dq, Kc, Vc, hd, kv_dim, gqa, d_pos, window, actq_for(q_dim));

        matmul_q(dout, wq_layer(m, L, "attn_output.weight"), dxb, q_dim, n_embd);
        rmsnorm_add_kernel<<<1, 256>>>(dx, dout, dW_layer(m, L, "post_attention_norm.weight"), n_embd, eps, NULL, AQ0);

        // ---- feed-forward (GeGLU) ----
        const int nff = m->ffn_len[L];
        rmsnorm_kernel<<<1, 256>>>(dh, dx, dW_layer(m, L, "ffn_norm.weight"), n_embd, eps, actq_for(n_embd));
        matmul_q(dg1, wq_layer(m, L, "ffn_gate.weight"), dh, n_embd, nff);
        matmul_q(dg2, wq_layer(m, L, "ffn_up.weight"), dh, n_embd, nff);          // dh still quantized
        geglu_kernel<<<gridn(nff), 256>>>(dg1, dg2, nff, actq_for(nff));
        matmul_q(dout, wq_layer(m, L, "ffn_down.weight"), dg1, nff, n_embd);
        rmsnorm_add_kernel<<<1, 256>>>(dx, dout, dW_layer(m, L, "post_ffw_norm.weight"), n_embd, eps,
                                       has_ple ? NULL : os, has_ple ? actq_for(n_embd) : AQ0);  // PLE reads dx next

        // ---- per-layer input (PLE) ----
        if (has_ple) {
            const int ple = c->n_embd_per_layer;
            matmul_q(dpg, wq_layer(m, L, "inp_gate.weight"), dx, n_embd, ple);
            geglu_kernel<<<gridn(ple), 256>>>(dpg, d_ipl + (size_t)L * ple, ple, actq_for(ple));
            matmul_q(dout, wq_layer(m, L, "proj.weight"), dpg, ple, n_embd);
            rmsnorm_add_kernel<<<1, 256>>>(dx, dout, dW_layer(m, L, "post_norm.weight"), n_embd, eps, os, AQ0);
        }
    }

    rmsnorm_kernel<<<1, 256>>>(dx, dx, dW(m, "output_norm.weight"), n_embd, eps, actq_for(n_embd));
    matmul_q(dlogits, wq(m, "token_embd.weight"), dx, n_embd, c->n_vocab);
    if (c->logit_softcap > 0.0f)
        softcap_kernel<<<gridn(c->n_vocab), 256>>>(dlogits, c->logit_softcap, c->n_vocab);
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
static cudaGraphExec_t g_graph_exec = NULL;
static int g_graph_warmups = 0;
static void forward_graph(struct model *m, struct kvcache *kv) {
    if (g_graph_warmups < 2) { g_graph_warmups++; forward_layers_and_head(m, kv); return; }
    if (!g_graph_exec) {
        cudaGraph_t graph;
        CUDA_CHECK(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        forward_layers_and_head(m, kv);
        CUDA_CHECK(cudaStreamEndCapture(cudaStreamPerThread, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&g_graph_exec, graph, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    CUDA_CHECK(cudaGraphLaunch(g_graph_exec, cudaStreamPerThread));
}

extern "C" void model_forward(struct model *m, struct kvcache *kv, int token, int pos, float *logits) {
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
    forward_graph(m, kv);               // capture once, then replay the forward each token

    CUDA_CHECK(cudaMemcpy(logits, dlogits, (size_t)c->n_vocab * 4, cudaMemcpyDeviceToHost));
}

#endif // MODEL_CUDA_CUH
