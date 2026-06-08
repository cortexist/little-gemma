// CUDA compute backend — milestone 1: matmul_q runs on the GPU (the quantized
// weight blob lives in VRAM; a kernel unpacks each row and dots it). The other
// ops stay on the host for now, matching model-cpu.c so results can be compared.
// Same model.h interface as the CPU backend; model.c provides the host setup.

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

// ====================  device-side dequantization  =========================
// One block per call, mirroring the host kernels in quant.c.

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

__device__ static void d_gsm(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4); }
}

__device__ static void d_dq_q4_K(const block_q4_K *x, float *y) {
    float d = d_fp16(x->d), mn = d_fp16(x->dmin);
    const uint8_t *q = x->qs; int is = 0;
    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc, m; d_gsm(is + 0, x->scales, &sc, &m); float d1 = d * sc, m1 = mn * m;
        d_gsm(is + 1, x->scales, &sc, &m); float d2 = d * sc, m2 = mn * m;
        for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4)  - m2;
        q += 32; is += 2;
    }
}
__device__ static void d_dq_q5_K(const block_q5_K *x, float *y) {
    float d = d_fp16(x->d), mn = d_fp16(x->dmin);
    const uint8_t *ql = x->qs, *qh = x->qh; int is = 0; uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc, m; d_gsm(is + 0, x->scales, &sc, &m); float d1 = d * sc, m1 = mn * m;
        d_gsm(is + 1, x->scales, &sc, &m); float d2 = d * sc, m2 = mn * m;
        for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4)  + ((qh[l] & u2) ? 16 : 0)) - m2;
        ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
    }
}
__device__ static void d_dq_q3_K(const block_q3_K *x, float *y) {
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    uint32_t aux[4]; const int8_t *scales = (const int8_t *)aux;
    float d_all = d_fp16(x->d); const uint8_t *q = x->qs, *hm = x->hmask; uint8_t m = 1;
    memcpy(aux, x->scales, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = ((aux[0] >> 0) & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = ((aux[1] >> 0) & kmask2) | (((tmp >> 2) & kmask1) << 4);
    int is = 0;
    for (int n = 0; n < QK_K; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            float dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
            dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
            shift += 2; m <<= 1;
        }
        q += 32;
    }
}
__device__ static void d_dq_q6_K(const block_q6_K *x, float *y) {
    float d = d_fp16(x->d); const uint8_t *ql = x->ql, *qh = x->qh; const int8_t *sc = x->scales;
    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;
            int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int8_t q3 = (int8_t)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int8_t q4 = (int8_t)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l +  0] = d * sc[is + 0] * q1; y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3; y[l + 96] = d * sc[is + 6] * q4;
        }
        y += 128; ql += 64; qh += 32; sc += 8;
    }
}
__device__ static void d_dq_q8_0(const block_q8_0 *x, float *y) {
    float d = d_fp16(x->d);
    for (int l = 0; l < 32; ++l) *y++ = d * x->qs[l];
}

__device__ static void d_dequant_block(int type, const unsigned char *src, float *buf) {
    switch (type) {
        case GGML_TYPE_F32:  buf[0] = *(const float *)src; break;
        case GGML_TYPE_F16:  buf[0] = d_fp16(*(const uint16_t *)src); break;
        case GGML_TYPE_BF16: buf[0] = d_bf16(*(const uint16_t *)src); break;
        case GGML_TYPE_Q8_0: d_dq_q8_0((const block_q8_0 *)src, buf); break;
        case GGML_TYPE_Q3_K: d_dq_q3_K((const block_q3_K *)src, buf); break;
        case GGML_TYPE_Q4_K: d_dq_q4_K((const block_q4_K *)src, buf); break;
        case GGML_TYPE_Q5_K: d_dq_q5_K((const block_q5_K *)src, buf); break;
        case GGML_TYPE_Q6_K: d_dq_q6_K((const block_q6_K *)src, buf); break;
    }
}

// ====================  matmul kernel + host wrapper  =======================

// One thread per output row: dequantize the row block-by-block, dot with x.
__global__ static void matmul_q_kernel(float *out, const unsigned char *wbase,
                                       int type, int ts, int blck, const float *x, int k, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const unsigned char *row = wbase + (size_t)i * (k / blck) * ts;
    float buf[QK_K];
    float s = 0.0f;
    int nb = k / blck;
    for (int b = 0; b < nb; b++) {
        d_dequant_block(type, row + (size_t)b * ts, buf);
        const float *xb = x + b * blck;
        for (int j = 0; j < blck; j++) s += buf[j] * xb[j];
    }
    out[i] = s;
}

// Device copy of the quantized weight blob, plus reusable activation buffers.
static const struct gguf_context *g_ctx = NULL;
static unsigned char *d_blob = NULL;
static float *d_x = NULL, *d_out = NULL;
static int d_x_cap = 0, d_out_cap = 0;

static void ensure_weights(struct model *m) {
    if (d_blob) return;
    g_ctx = m->ctx;
    CUDA_CHECK(cudaMalloc(&d_blob, m->ctx->data_size));
    CUDA_CHECK(cudaMemcpy(d_blob, m->ctx->data, m->ctx->data_size, cudaMemcpyHostToDevice));
}

static const unsigned char *dev_weight(const struct gguf_tensor *t) {
    size_t off = (const unsigned char *)t->data - (const unsigned char *)g_ctx->data;
    return d_blob + off;
}

static void matmul_q(float *out, const struct gguf_tensor *t, const float *x, int k, int m) {
    if (k > d_x_cap)   { cudaFree(d_x);   CUDA_CHECK(cudaMalloc(&d_x,   (size_t)k * sizeof(float))); d_x_cap = k; }
    if (m > d_out_cap) { cudaFree(d_out); CUDA_CHECK(cudaMalloc(&d_out, (size_t)m * sizeof(float))); d_out_cap = m; }
    CUDA_CHECK(cudaMemcpy(d_x, x, (size_t)k * sizeof(float), cudaMemcpyHostToDevice));
    int blck = ggml_blck_size(t->type), ts = (int)ggml_type_size(t->type);
    int threads = 256, blocks = (m + threads - 1) / threads;
    matmul_q_kernel<<<blocks, threads>>>(d_out, dev_weight(t), (int)t->type, ts, blck, d_x, k, m);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out, d_out, (size_t)m * sizeof(float), cudaMemcpyDeviceToHost));
}

// ====================  host kernels (unchanged from CPU)  ===================

static void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    float ss = 0.0f; for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float s = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * s * w[i];
}
static void rope_neox(float *v, int d, int pos, float base, const float *ff) {
    int half = d / 2;
    for (int i = 0; i < half; i++) {
        float freq = powf(base, -2.0f * (float)i / (float)d);
        float ang = (float)pos * freq / (ff ? ff[i] : 1.0f);
        float c = cosf(ang), s = sinf(ang), a = v[i], b = v[i + half];
        v[i] = a * c - b * s; v[i + half] = a * s + b * c;
    }
}
static void rmsnorm_plain(float *out, const float *x, int n, float eps) {
    float ss = 0.0f; for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float s = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * s;
}
static float gelu(float x) {
    const float k = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}
static void softmax(float *x, int n) {
    float mx = x[0]; for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f; for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// ====================  weight access / geometry (host)  =====================

static const struct gguf_tensor *wq(struct model *m, const char *name) { return gguf_find_tensor(m->ctx, name); }
static const struct gguf_tensor *wq_layer(struct model *m, int L, const char *suffix) {
    char name[96]; snprintf(name, sizeof name, "blk.%d.%s", L, suffix);
    return gguf_find_tensor(m->ctx, name);
}
static const float *fptr(struct model *m, const char *name) {
    const struct gguf_tensor *t = gguf_find_tensor(m->ctx, name);
    return t ? (const float *)t->data : NULL;
}
static const float *fptr_layer(struct model *m, int L, const char *suffix) {
    char name[96]; snprintf(name, sizeof name, "blk.%d.%s", L, suffix);
    return fptr(m, name);
}
static int head_dim_at(const struct model *m, int L) { return m->head_dim[L]; }
static int kv_src(const struct model *m, int L) {
    const struct config *c = &m->cfg;
    if (L < c->n_kv_start) return L;
    return c->n_kv_start - (m->is_local[L] ? 2 : 1);
}

// ====================  kv cache (host buffers)  =============================

extern "C" int kvcache_init(struct kvcache *kv, const struct model *m, int max_seq) {
    const struct config *c = &m->cfg;
    kv->n_layer = c->n_layer; kv->max_seq = max_seq;
    kv->kv_dim = (int *)calloc((size_t)c->n_layer, sizeof(int));
    kv->k = (float **)calloc((size_t)c->n_layer, sizeof(float *));
    kv->v = (float **)calloc((size_t)c->n_layer, sizeof(float *));
    if (!kv->kv_dim || !kv->k || !kv->v) return -1;
    for (int L = 0; L < c->n_layer; L++) {
        kv->kv_dim[L] = m->n_head_kv[L] * head_dim_at(m, L);
        size_t n = (size_t)max_seq * kv->kv_dim[L];
        kv->k[L] = (float *)calloc(n, sizeof(float));
        kv->v[L] = (float *)calloc(n, sizeof(float));
        if (!kv->k[L] || !kv->v[L]) return -1;
    }
    return 0;
}
extern "C" void kvcache_free(struct kvcache *kv) {
    if (!kv) return;
    for (int L = 0; L < kv->n_layer; L++) { free(kv->k[L]); free(kv->v[L]); }
    free(kv->k); free(kv->v); free(kv->kv_dim);
    kv->k = kv->v = NULL; kv->kv_dim = NULL;
}

// ====================  forward pass  =======================================

static float *build_per_layer(struct model *m, int token, const float *inp_scaled) {
    const struct config *c = &m->cfg;
    const int ple = c->n_embd_per_layer;
    if (ple <= 0) return NULL;
    const struct gguf_tensor *pte = gguf_find_tensor(m->ctx, "per_layer_token_embd.weight");
    if (!pte) return NULL;
    const int64_t total = (int64_t)ple * c->n_layer;

    float *tok = dequantize_row(pte, token, total);
    if (!tok) return NULL;
    float te_scale = sqrtf((float)ple);
    for (int64_t i = 0; i < total; i++) tok[i] *= te_scale;

    float *proj = (float *)malloc((size_t)total * sizeof(float));
    matmul_q(proj, wq(m, "per_layer_model_proj.weight"), inp_scaled, c->n_embd, (int)total);
    float pscale = 1.0f / sqrtf((float)c->n_embd);
    for (int64_t i = 0; i < total; i++) proj[i] *= pscale;

    const float *pn = fptr(m, "per_layer_proj_norm.weight");
    for (int L = 0; L < c->n_layer; L++) rmsnorm(proj + L * ple, proj + L * ple, pn, ple, c->rms_eps);
    float inv_sqrt2 = 1.0f / sqrtf(2.0f);
    for (int64_t i = 0; i < total; i++) proj[i] = (proj[i] + tok[i]) * inv_sqrt2;
    free(tok);
    return proj;
}

extern "C" void model_forward(struct model *m, struct kvcache *kv, int token, int pos, float *logits) {
    ensure_weights(m);
    const struct config *c = &m->cfg;
    const int n_embd = c->n_embd, n_head = c->n_head;
    const int n_ff = c->n_ff, ple = c->n_embd_per_layer;
    const float eps = c->rms_eps;

    int maxhd = 0, max_kvdim = 0;
    for (int L = 0; L < c->n_layer; L++) {
        if (m->head_dim[L] > maxhd) maxhd = m->head_dim[L];
        int kvd = m->n_head_kv[L] * m->head_dim[L];
        if (kvd > max_kvdim) max_kvdim = kvd;
    }
    const int q_max = n_head * maxhd, kv_max = max_kvdim;

    float *x   = (float *)malloc((size_t)n_embd * sizeof(float));
    float *h   = (float *)malloc((size_t)n_embd * sizeof(float));
    float *q   = (float *)malloc((size_t)q_max  * sizeof(float));
    float *kb  = (float *)malloc((size_t)kv_max * sizeof(float));
    float *vb  = (float *)malloc((size_t)kv_max * sizeof(float));
    float *xb  = (float *)malloc((size_t)q_max  * sizeof(float));
    float *o   = (float *)malloc((size_t)n_embd * sizeof(float));
    float *g1  = (float *)malloc((size_t)n_ff   * sizeof(float));
    float *g2  = (float *)malloc((size_t)n_ff   * sizeof(float));
    float *att = (float *)malloc((size_t)(pos + 1) * sizeof(float));
    float *pg  = ple > 0 ? (float *)malloc((size_t)ple * sizeof(float)) : NULL;

    float *erow = dequantize_row(wq(m, "token_embd.weight"), token, n_embd);
    for (int i = 0; i < n_embd; i++) x[i] = erow[i] * sqrtf((float)n_embd);
    free(erow);

    float *inp_per_layer = build_per_layer(m, token, x);
    const float *rope_freqs = fptr(m, "rope_freqs.weight");

    for (int L = 0; L < c->n_layer; L++) {
        const int local = m->is_local[L];
        const int hd = head_dim_at(m, L);
        const int n_head_kv = m->n_head_kv[L];
        const int q_dim = n_head * hd, kv_dim = n_head_kv * hd;
        const float base = local ? c->rope_freq_base_swa : c->rope_freq_base;
        const float *ff = local ? NULL : rope_freqs;

        rmsnorm(h, x, fptr_layer(m, L, "attn_norm.weight"), n_embd, eps);
        matmul_q(q, wq_layer(m, L, "attn_q.weight"), h, n_embd, q_dim);
        const float *qn = fptr_layer(m, L, "attn_q_norm.weight");
        for (int hh = 0; hh < n_head; hh++) rmsnorm(q + hh * hd, q + hh * hd, qn, hd, eps);
        for (int hh = 0; hh < n_head; hh++) rope_neox(q + hh * hd, hd, pos, base, ff);

        int src = kv_src(m, L);
        if (L < c->n_kv_start) {
            matmul_q(kb, wq_layer(m, L, "attn_k.weight"), h, n_embd, kv_dim);
            const struct gguf_tensor *wv = wq_layer(m, L, "attn_v.weight");
            if (wv) matmul_q(vb, wv, h, n_embd, kv_dim);
            else    memcpy(vb, kb, (size_t)kv_dim * sizeof(float));
            const float *kn = fptr_layer(m, L, "attn_k_norm.weight");
            for (int hh = 0; hh < n_head_kv; hh++) rmsnorm(kb + hh * hd, kb + hh * hd, kn, hd, eps);
            for (int hh = 0; hh < n_head_kv; hh++) rmsnorm_plain(vb + hh * hd, vb + hh * hd, hd, eps);
            for (int hh = 0; hh < n_head_kv; hh++) rope_neox(kb + hh * hd, hd, pos, base, ff);
            memcpy(kv->k[L] + (size_t)pos * kv_dim, kb, (size_t)kv_dim * sizeof(float));
            memcpy(kv->v[L] + (size_t)pos * kv_dim, vb, (size_t)kv_dim * sizeof(float));
        }
        const float *Kc = kv->k[src], *Vc = kv->v[src];

        int start = (local && c->sliding_window > 0 && pos - c->sliding_window + 1 > 0)
                  ? pos - c->sliding_window + 1 : 0;
        const int gqa = n_head / n_head_kv;
        for (int hh = 0; hh < n_head; hh++) {
            const float *qh = q + hh * hd; int kvh = hh / gqa;
            for (int t = start; t <= pos; t++) {
                const float *kt = Kc + (size_t)t * kv_dim + (size_t)kvh * hd;
                float s = 0.0f; for (int i = 0; i < hd; i++) s += qh[i] * kt[i];
                att[t] = s;
            }
            softmax(att + start, pos - start + 1);
            float *outh = xb + hh * hd;
            for (int i = 0; i < hd; i++) outh[i] = 0.0f;
            for (int t = start; t <= pos; t++) {
                const float *vt = Vc + (size_t)t * kv_dim + (size_t)kvh * hd;
                float a = att[t]; for (int i = 0; i < hd; i++) outh[i] += a * vt[i];
            }
        }

        matmul_q(o, wq_layer(m, L, "attn_output.weight"), xb, q_dim, n_embd);
        rmsnorm(o, o, fptr_layer(m, L, "post_attention_norm.weight"), n_embd, eps);
        for (int i = 0; i < n_embd; i++) x[i] += o[i];

        const int nff = m->ffn_len[L];
        rmsnorm(h, x, fptr_layer(m, L, "ffn_norm.weight"), n_embd, eps);
        matmul_q(g1, wq_layer(m, L, "ffn_gate.weight"), h, n_embd, nff);
        matmul_q(g2, wq_layer(m, L, "ffn_up.weight"),   h, n_embd, nff);
        for (int i = 0; i < nff; i++) g1[i] = gelu(g1[i]) * g2[i];
        matmul_q(o, wq_layer(m, L, "ffn_down.weight"), g1, nff, n_embd);
        rmsnorm(o, o, fptr_layer(m, L, "post_ffw_norm.weight"), n_embd, eps);
        for (int i = 0; i < n_embd; i++) x[i] += o[i];

        if (inp_per_layer) {
            const float *ile = inp_per_layer + (size_t)L * ple;
            matmul_q(pg, wq_layer(m, L, "inp_gate.weight"), x, n_embd, ple);
            for (int i = 0; i < ple; i++) pg[i] = gelu(pg[i]) * ile[i];
            matmul_q(o, wq_layer(m, L, "proj.weight"), pg, ple, n_embd);
            rmsnorm(o, o, fptr_layer(m, L, "post_norm.weight"), n_embd, eps);
            for (int i = 0; i < n_embd; i++) x[i] += o[i];
        }

        const float *os = fptr_layer(m, L, "layer_output_scale.weight");
        if (os) { for (int i = 0; i < n_embd; i++) x[i] *= os[0]; }
    }

    rmsnorm(x, x, fptr(m, "output_norm.weight"), n_embd, eps);
    matmul_q(logits, wq(m, "token_embd.weight"), x, n_embd, c->n_vocab);
    if (c->logit_softcap > 0.0f) {
        float sc = c->logit_softcap;
        for (int v = 0; v < c->n_vocab; v++) logits[v] = sc * tanhf(logits[v] / sc);
    }

    free(x); free(h); free(q); free(kb); free(vb); free(xb);
    free(o); free(g1); free(g2); free(att); free(pg); free(inp_per_layer);
}
