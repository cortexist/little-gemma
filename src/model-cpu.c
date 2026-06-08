// CPU backend: the compute kernels and the forward pass. The host-side setup
// (config, per-layer geometry) lives in model.c; the CUDA backend (model-cuda.cu)
// is an alternative to this file behind the same model.h interface.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "model.h"
#include "quant.h"

// ---- math kernels (plain f32; the part CUDA will replace) -----------------

// out[i] = x[i] / rms(x) * w[i]. Safe in place (out == x).
static void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float s = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * s * w[i];
}

// GPT-NeoX rotary embedding on one head vector of size d, at position pos.
// `ff` (or NULL) holds per-pair frequency divisors (rope_freqs) for global layers.
static void rope_neox(float *v, int d, int pos, float base, const float *ff) {
    int half = d / 2;
    for (int i = 0; i < half; i++) {
        float freq = powf(base, -2.0f * (float)i / (float)d);
        float ang = (float)pos * freq / (ff ? ff[i] : 1.0f);
        float c = cosf(ang), s = sinf(ang);
        float a = v[i], b = v[i + half];
        v[i]        = a * c - b * s;
        v[i + half] = a * s + b * c;
    }
}

// Plain RMS normalization with no learned weight (Gemma normalizes V this way).
static void rmsnorm_plain(float *out, const float *x, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float s = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * s;
}

// GELU (tanh approximation), as Gemma uses for the FFN gate.
static float gelu(float x) {
    const float k = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

static void softmax(float *x, int n) {
    float max = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// ---- weight access --------------------------------------------------------

// A quantized weight tensor, by name / by layer-and-role.
static const struct gguf_tensor *wq(struct model *m, const char *name) {
    return gguf_find_tensor(m->ctx, name);
}
static const struct gguf_tensor *wq_layer(struct model *m, int L, const char *suffix) {
    char name[96];
    snprintf(name, sizeof name, "blk.%d.%s", L, suffix);
    return gguf_find_tensor(m->ctx, name);
}

// Norm/scale weights are stored as f32, so point straight at them (zero copy).
static const float *fptr(struct model *m, const char *name) {
    const struct gguf_tensor *t = gguf_find_tensor(m->ctx, name);
    return t ? (const float *)t->data : NULL;
}
static const float *fptr_layer(struct model *m, int L, const char *suffix) {
    char name[96];
    snprintf(name, sizeof name, "blk.%d.%s", L, suffix);
    return fptr(m, name);
}

// out[m] = W . x with W quantized: unpack each row on the fly (no f32 weight
// copy kept in memory). Reads only the quantized bytes per token. Rows are
// independent, so this threads across cores like the f32 matmul.
static void matmul_q(float *out, const struct gguf_tensor *t, const float *x, int k, int m) {
    const int blck = ggml_blck_size(t->type);
    const size_t row_bytes = (size_t)(k / blck) * ggml_type_size(t->type);
    const unsigned char *base = t->data;
    #pragma omp parallel
    {
        float *buf = malloc((size_t)k * sizeof(float));   // one row, reused
        int i;
        #pragma omp for schedule(static)
        for (i = 0; i < m; i++) {
            dequantize_into(t->type, base + (size_t)i * row_bytes, buf, k);
            float s = 0.0f;
            for (int j = 0; j < k; j++) s += buf[j] * x[j];
            out[i] = s;
        }
        free(buf);
    }
}

// ---- per-layer geometry helpers (shared shape, set up in model.c) ----------

// Head size for layer L (derived from the q-projection at load time).
static int head_dim_at(const struct model *m, int L) {
    return m->head_dim[L];
}

// Which layer's KV layer L uses. Layers [0,n_kv_start) own theirs; later layers
// reuse the last same-type KV layer (local -> n_kv_start-2, global -> n_kv_start-1).
static int kv_src(const struct model *m, int L) {
    const struct config *c = &m->cfg;
    if (L < c->n_kv_start) return L;
    return c->n_kv_start - (m->is_local[L] ? 2 : 1);
}

// ---- kv cache (host buffers) ----------------------------------------------

int kvcache_init(struct kvcache *kv, const struct model *m, int max_seq) {
    const struct config *c = &m->cfg;
    kv->n_layer = c->n_layer;
    kv->max_seq = max_seq;
    kv->kv_dim = calloc((size_t)c->n_layer, sizeof(int));
    kv->k = calloc((size_t)c->n_layer, sizeof(float *));
    kv->v = calloc((size_t)c->n_layer, sizeof(float *));
    if (!kv->kv_dim || !kv->k || !kv->v) return -1;

    for (int L = 0; L < c->n_layer; L++) {
        kv->kv_dim[L] = m->n_head_kv[L] * head_dim_at(m, L);
        size_t n = (size_t)max_seq * kv->kv_dim[L];
        kv->k[L] = calloc(n, sizeof(float));
        kv->v[L] = calloc(n, sizeof(float));
        if (!kv->k[L] || !kv->v[L]) return -1;
    }
    return 0;
}

void kvcache_free(struct kvcache *kv) {
    if (!kv) return;
    for (int L = 0; L < kv->n_layer; L++) { free(kv->k[L]); free(kv->v[L]); }
    free(kv->k); free(kv->v); free(kv->kv_dim);
    kv->k = kv->v = NULL; kv->kv_dim = NULL;
}

// ---- the forward pass -----------------------------------------------------

// Build the per-layer input vectors (PLE): one n_embd_per_layer slice per layer.
// Returns a malloc'd [n_embd_per_layer * n_layer] array, or NULL if absent.
// inp_scaled is the sqrt-scaled token embedding (the model's layer-0 input).
static float *build_per_layer(struct model *m, int token, const float *inp_scaled) {
    const struct config *c = &m->cfg;
    const int ple = c->n_embd_per_layer;
    if (ple <= 0) return NULL;
    const struct gguf_tensor *pte = gguf_find_tensor(m->ctx, "per_layer_token_embd.weight");
    if (!pte) return NULL;
    const int64_t total = (int64_t)ple * c->n_layer;

    // token's per-layer embedding row, scaled by sqrt(n_embd_per_layer)
    float *tok = dequantize_row(pte, token, total);   // per token, not cached
    if (!tok) return NULL;
    float te_scale = sqrtf((float)ple);
    for (int64_t i = 0; i < total; i++) tok[i] *= te_scale;

    // project the main embedding: per_layer_model_proj @ inp_scaled, scaled by 1/sqrt(n_embd)
    float *proj = malloc((size_t)total * sizeof(float));
    matmul_q(proj, wq(m, "per_layer_model_proj.weight"), inp_scaled, c->n_embd, (int)total);
    float pscale = 1.0f / sqrtf((float)c->n_embd);
    for (int64_t i = 0; i < total; i++) proj[i] *= pscale;

    // RMS normalization each per-layer slice, then combine: (proj + tok) / sqrt(2)
    const float *pn = fptr(m, "per_layer_proj_norm.weight");
    for (int L = 0; L < c->n_layer; L++) rmsnorm(proj + L * ple, proj + L * ple, pn, ple, c->rms_eps);
    float inv_sqrt2 = 1.0f / sqrtf(2.0f);
    for (int64_t i = 0; i < total; i++) proj[i] = (proj[i] + tok[i]) * inv_sqrt2;
    free(tok);
    return proj; // [ple, n_layer], slice for layer L at proj + L*ple
}

void model_forward(struct model *m, struct kvcache *kv, int token, int pos, float *logits) {
    const struct config *c = &m->cfg;
    const int n_embd = c->n_embd, n_head = c->n_head;
    const int n_ff = c->n_ff, ple = c->n_embd_per_layer;
    const float eps = c->rms_eps;

    // scratch sized for the widest layer (head_dim and n_head_kv vary per layer)
    int maxhd = 0, max_kvdim = 0;
    for (int L = 0; L < c->n_layer; L++) {
        if (m->head_dim[L] > maxhd) maxhd = m->head_dim[L];
        int kvd = m->n_head_kv[L] * m->head_dim[L];
        if (kvd > max_kvdim) max_kvdim = kvd;
    }
    const int q_max  = n_head * maxhd;
    const int kv_max = max_kvdim;

    // scratch (sized for the largest layer)
    float *x   = malloc((size_t)n_embd * sizeof(float));
    float *h   = malloc((size_t)n_embd * sizeof(float));
    float *q   = malloc((size_t)q_max  * sizeof(float));
    float *kb  = malloc((size_t)kv_max * sizeof(float));
    float *vb  = malloc((size_t)kv_max * sizeof(float));
    float *xb  = malloc((size_t)q_max  * sizeof(float));
    float *o   = malloc((size_t)n_embd * sizeof(float));
    float *g1  = malloc((size_t)n_ff   * sizeof(float));
    float *g2  = malloc((size_t)n_ff   * sizeof(float));
    float *att = malloc((size_t)(pos + 1) * sizeof(float));
    float *pg  = ple > 0 ? malloc((size_t)ple * sizeof(float)) : NULL;

    // embedding lookup (one quantized row), scaled by sqrt(n_embd)
    float *erow = dequantize_row(wq(m, "token_embd.weight"), token, n_embd);
    for (int i = 0; i < n_embd; i++) x[i] = erow[i] * sqrtf((float)n_embd);
    free(erow);

    // per-layer inputs (uses the scaled embedding above)
    float *inp_per_layer = build_per_layer(m, token, x);

    // rope_freqs (freq_factors, stored f32) used by global/full layers only
    const float *rope_freqs = fptr(m, "rope_freqs.weight");

    for (int L = 0; L < c->n_layer; L++) {
        const int local = m->is_local[L];
        const int hd = head_dim_at(m, L);
        const int n_head_kv = m->n_head_kv[L];
        const int q_dim = n_head * hd, kv_dim = n_head_kv * hd;
        const float base = local ? c->rope_freq_base_swa : c->rope_freq_base;
        const float *ff = local ? NULL : rope_freqs;

        // ---- attention ----
        rmsnorm(h, x, fptr_layer(m, L, "attn_norm.weight"), n_embd, eps);

        matmul_q(q, wq_layer(m, L, "attn_q.weight"), h, n_embd, q_dim);
        const float *qn = fptr_layer(m, L, "attn_q_norm.weight");
        for (int hh = 0; hh < n_head; hh++) rmsnorm(q + hh * hd, q + hh * hd, qn, hd, eps);
        for (int hh = 0; hh < n_head; hh++) rope_neox(q + hh * hd, hd, pos, base, ff);

        int src = kv_src(m, L);
        if (L < c->n_kv_start) {                 // this layer owns its KV
            matmul_q(kb, wq_layer(m, L, "attn_k.weight"), h, n_embd, kv_dim);
            const struct gguf_tensor *wv = wq_layer(m, L, "attn_v.weight");
            if (wv) matmul_q(vb, wv, h, n_embd, kv_dim);
            else    memcpy(vb, kb, (size_t)kv_dim * sizeof(float)); // no V proj: V = K projection
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
            const float *qh = q + hh * hd;
            int kvh = hh / gqa;
            for (int t = start; t <= pos; t++) {
                const float *kt = Kc + (size_t)t * kv_dim + (size_t)kvh * hd;
                float s = 0.0f;
                for (int i = 0; i < hd; i++) s += qh[i] * kt[i];
                att[t] = s; // Gemma4 attention scale is 1.0 (no 1/sqrt(d))
            }
            softmax(att + start, pos - start + 1);
            float *outh = xb + hh * hd;
            for (int i = 0; i < hd; i++) outh[i] = 0.0f;
            for (int t = start; t <= pos; t++) {
                const float *vt = Vc + (size_t)t * kv_dim + (size_t)kvh * hd;
                float a = att[t];
                for (int i = 0; i < hd; i++) outh[i] += a * vt[i];
            }
        }

        matmul_q(o, wq_layer(m, L, "attn_output.weight"), xb, q_dim, n_embd);
        rmsnorm(o, o, fptr_layer(m, L, "post_attention_norm.weight"), n_embd, eps);
        for (int i = 0; i < n_embd; i++) x[i] += o[i];   // attn residual -> attn_out

        // ---- feed-forward (GeGLU); width is per-layer (elastic FFN) ----
        const int nff = m->ffn_len[L];
        rmsnorm(h, x, fptr_layer(m, L, "ffn_norm.weight"), n_embd, eps);
        matmul_q(g1, wq_layer(m, L, "ffn_gate.weight"), h, n_embd, nff);
        matmul_q(g2, wq_layer(m, L, "ffn_up.weight"),   h, n_embd, nff);
        for (int i = 0; i < nff; i++) g1[i] = gelu(g1[i]) * g2[i];
        matmul_q(o, wq_layer(m, L, "ffn_down.weight"), g1, nff, n_embd);
        rmsnorm(o, o, fptr_layer(m, L, "post_ffw_norm.weight"), n_embd, eps);
        for (int i = 0; i < n_embd; i++) x[i] += o[i];   // ffn residual

        // ---- per-layer input (PLE) ----
        if (inp_per_layer) {
            const float *ile = inp_per_layer + (size_t)L * ple;
            matmul_q(pg, wq_layer(m, L, "inp_gate.weight"), x, n_embd, ple);
            for (int i = 0; i < ple; i++) pg[i] = gelu(pg[i]) * ile[i];
            matmul_q(o, wq_layer(m, L, "proj.weight"), pg, ple, n_embd);
            rmsnorm(o, o, fptr_layer(m, L, "post_norm.weight"), n_embd, eps);
            for (int i = 0; i < n_embd; i++) x[i] += o[i];   // PLE residual
        }

        // ---- per-layer output scale (f32 scalar) ----
        const float *os = fptr_layer(m, L, "layer_output_scale.weight");
        if (os) { for (int i = 0; i < n_embd; i++) x[i] *= os[0]; }
    }

    // final norm + tied output projection (logits = token_embd . x), then softcap
    rmsnorm(x, x, fptr(m, "output_norm.weight"), n_embd, eps);
    matmul_q(logits, wq(m, "token_embd.weight"), x, n_embd, c->n_vocab);
    if (c->logit_softcap > 0.0f) {
        float sc = c->logit_softcap;
        for (int v = 0; v < c->n_vocab; v++) logits[v] = sc * tanhf(logits[v] / sc);
    }

    free(x); free(h); free(q); free(kb); free(vb); free(xb);
    free(o); free(g1); free(g2); free(att); free(pg);
    free(inp_per_layer);   // rope_freqs and all weights are owned by the cache
}
