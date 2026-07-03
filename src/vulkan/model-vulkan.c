// Vulkan backend: model-cpu.c's readable forward with the quantized matmul —
// nearly all of a token's weight traffic and compute — dispatched to the GPU
// (vk-compute.c, one compute pipeline per weight type). Everything else
// (norms, RoPE, attention over the host KV cache, GELU) stays the CPU
// reference code: on the integrated GPUs this backend targets, device memory
// IS system RAM, so the activation handoff each way is a memcpy, not a bus.
// The CUDA journey ran the same route — correctness first, then move the
// forward across piece by measured piece.
//
// LG_VK_VERIFY=1 runs every GPU matmul beside the host oracle and reports the
// max difference (the LG_MEDIA_VERIFY pattern): expected ~1e-5-scale, pure
// summation-order reassociation — the dequantized values are bit-identical.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "model.h"
#include "quant.h"
#include "vulkan/vk-compute.h"

// ---- math kernels (host f32; the matmul below is the part the GPU takes) ---

const int model_kv_host = 1;     // this backend's kvcache rows are host memory

void kvcache_save_prefix(struct kvcache *kv, int n) {
    kv->px_k = calloc((size_t)kv->n_layer, sizeof(void *));
    kv->px_v = calloc((size_t)kv->n_layer, sizeof(void *));
    if (!kv->px_k || !kv->px_v) return;
    for (int L = 0; L < kv->n_layer; L++) {
        if (!kv->k[L]) continue;                       // reuse layer: nothing stored
        int rows = n < kv->seq[L] ? n : kv->seq[L];
        size_t bytes = (size_t)rows * kv->kv_dim[L] * (kv->f16[L] ? 2 : 4);
        kv->px_k[L] = malloc(bytes);
        kv->px_v[L] = malloc(bytes);
        if (!kv->px_k[L] || !kv->px_v[L]) return;
        memcpy(kv->px_k[L], kv->k[L], bytes);
        memcpy(kv->px_v[L], kv->v[L], bytes);
    }
}

void kvcache_restore_prefix(struct kvcache *kv, int n) {
    if (!kv->px_k) return;
    for (int L = 0; L < kv->n_layer; L++) {
        if (!kv->px_k[L]) continue;
        int rows = n < kv->seq[L] ? n : kv->seq[L];
        size_t bytes = (size_t)rows * kv->kv_dim[L] * (kv->f16[L] ? 2 : 4);
        memcpy(kv->k[L], kv->px_k[L], bytes);
        memcpy(kv->v[L], kv->px_v[L], bytes);
    }
}

// The MTP verify, sequentially — LG_MTP_N forwards, byte-identical to plain decode by
// construction (they ARE the same forwards). The next forward only runs while the
// previous draft held, so last_hidden lands on the last valid position automatically.
int model_forward_spec(struct model *m, struct kvcache *kv, const int *toks, int pos, int *out) {
    int j = 0;
    do {
        out[j] = model_forward_next(m, kv, toks[j], pos + j);
        j++;
    } while (j < LG_MTP_N && out[j - 1] == toks[j]);
    return j;
}

// The host draft path in mtp.c does all the work on this backend (the KV cache
// is host memory, exactly what it needs).
#include "mtp-internal.h"
int  mtp_draft_device(struct mtp *t, const struct model *m, const struct kvcache *kv,
                      int token, int pos) {
    (void)t; (void)m; (void)kv; (void)token; (void)pos;
    return -1;
}
int  mtp_draft_chain_device(struct mtp *t, const struct model *m, const struct kvcache *kv,
                            int token, int pos) {
    (void)t; (void)m; (void)kv; (void)token; (void)pos;
    return -1;
}
void mtp_free_device(struct mtp *t) { (void)t; }

// The device prefill buffers this presizes are a CUDA-graph concern; this
// backend sizes its I/O buffers on demand — no-op.
void model_prefill_reserve(void) { }

static void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float s = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * s * w[i];
}

// GPT-NeoX rotary embedding on one head vector of size d, at position pos.
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

// ---- weight access ----------------------------------------------------------

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

// The CPU reference matmul (model-cpu.c's matmul_q verbatim): the fallback for
// weight types without a shader, and the oracle under LG_VK_VERIFY.
static void matmul_host(float *out, const struct gguf_tensor *t, const float *x, int k, int m) {
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

// out[m] = W . x — on the GPU when a pipeline exists for the weight type,
// on the host otherwise (announced once per type, never silently).
static void matmul_q(float *out, const struct gguf_tensor *t, const float *x, int k, int m) {
    static int verify = -1;
    static uint32_t warned = 0;               // one bit per ggml type id
    if (verify < 0) verify = getenv("LG_VK_VERIFY") != NULL;

    if (vkc_matmul(out, t, x, k, m) != 0) {
        if (t->type < 32 && !(warned & (1u << t->type))) {
            warned |= 1u << t->type;
            fprintf(stderr, "vulkan: no %s pipeline — %s stays on the host matmul\n",
                    ggml_type_name(t->type), t->name);
        }
        matmul_host(out, t, x, k, m);
        return;
    }
    if (verify) {
        static float worst = 0.0f;
        static int calls = 0;
        float *ref = malloc((size_t)m * sizeof(float));
        if (!ref) return;
        matmul_host(ref, t, x, k, m);
        float md = 0.0f, scale = 0.0f;
        for (int i = 0; i < m; i++) {
            float d = fabsf(out[i] - ref[i]);
            if (d > md) md = d;
            float a = fabsf(ref[i]);
            if (a > scale) scale = a;
        }
        calls++;
        if (calls <= 16 || md > worst)
            fprintf(stderr, "vulkan verify[%d]: %-6s %6d x %-6d  max|gpu-host| %.3e  (max|host| %.3e)\n",
                    calls, ggml_type_name(t->type), k, m, md, scale);
        if (md > worst) worst = md;
        free(ref);
    }
}

// ---- per-layer geometry helpers (shared shape, set up in model.c) -----------

static int head_dim_at(const struct model *m, int L) {
    return m->head_dim[L];
}

// Which layer's KV layer L uses (layers >= n_kv_start reuse an earlier ring).
static int kv_src(const struct model *m, int L) {
    const struct config *c = &m->cfg;
    if (L < c->n_kv_start) return L;
    return c->n_kv_start - (m->is_local[L] ? 2 : 1);
}

// ---- f32 <-> f16 (for the f16-stored global-layer KV rows) ------------------

// Round-to-nearest-even, bit-exact with the CPU and CUDA backends — every
// backend must store the same rounded value or their outputs drift.
static uint16_t f16_of(float f) {
    float scale_to_inf, scale_to_zero;                  // 2^112, 2^-110
    { uint32_t b = 0x77800000u; memcpy(&scale_to_inf,  &b, 4); }
    { uint32_t b = 0x08800000u; memcpy(&scale_to_zero, &b, 4); }
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;
    uint32_t w; memcpy(&w, &f, 4);
    uint32_t shl1_w = w + w;
    uint32_t sign = w & 0x80000000u;
    uint32_t bias = shl1_w & 0xFF000000u;
    if (bias < 0x71000000u) bias = 0x71000000u;
    float fb; { uint32_t b = (bias >> 1) + 0x07800000u; memcpy(&fb, &b, 4); }
    base = fb + base;
    uint32_t bits; memcpy(&bits, &base, 4);
    uint32_t nonsign = ((bits >> 13) & 0x00007C00u) + (bits & 0x00000FFFu);
    return (uint16_t)((sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
}
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16, exp = (h >> 10) & 0x1Fu, mant = h & 0x3FFu, bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else { exp = 127 - 15 + 1; while ((mant & 0x400u) == 0) { mant <<= 1; exp--; } mant &= 0x3FFu; bits = sign | (exp << 23) | (mant << 13); }
    } else if (exp == 0x1F) bits = sign | 0x7F800000u | (mant << 13);
    else bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    float f; memcpy(&f, &bits, 4);
    return f;
}
// One element of a cache row, whichever way the layer stores it.
static float kv_at(const void *row, size_t i, int f16) {
    return f16 ? f16_to_f32(((const uint16_t *)row)[i]) : ((const float *)row)[i];
}

// ---- kv cache (host buffers) -------------------------------------------------

int kvcache_init(struct kvcache *kv, const struct model *m, int max_seq) {
    // First code that both holds the model AND runs before any forward: bring
    // Vulkan up here. A failure is loud but not fatal — every matmul then
    // takes the host path, which is just the CPU backend with extra steps.
    vkc_init(m->ctx);

    const struct config *c = &m->cfg;
    kv->n_layer = c->n_layer;
    kv->max_seq = max_seq;
    kv->px_k = kv->px_v = NULL;
    kv->kv_dim = calloc((size_t)c->n_layer, sizeof(int));
    kv->seq = calloc((size_t)c->n_layer, sizeof(int));
    kv->f16 = calloc((size_t)c->n_layer, sizeof(int));
    kv->k = calloc((size_t)c->n_layer, sizeof(void *));
    kv->v = calloc((size_t)c->n_layer, sizeof(void *));
    if (!kv->kv_dim || !kv->seq || !kv->f16 || !kv->k || !kv->v) return -1;

    for (int L = 0; L < c->n_layer; L++) {
        kv->kv_dim[L] = m->n_head_kv[L] * head_dim_at(m, L);
        if (L >= c->n_kv_start) continue;     // reuses kv_src's buffers: k/v stay NULL
        // Sliding-window layers keep a ring of window rows; single-token
        // forwards write p % seq over the row that just slid out of reach.
        int seq = max_seq;
        if (m->is_local[L] && c->sliding_window > 0 && c->sliding_window < max_seq)
            seq = c->sliding_window;
        kv->seq[L] = seq;
        kv->f16[L] = !m->is_local[L];         // global rows are f16 (see model.h)
        size_t n = (size_t)seq * kv->kv_dim[L];
        kv->k[L] = calloc(n, kv->f16[L] ? 2 : 4);
        kv->v[L] = calloc(n, kv->f16[L] ? 2 : 4);
        if (!kv->k[L] || !kv->v[L]) return -1;
    }
    return 0;
}

void kvcache_free(struct kvcache *kv) {
    if (!kv) return;
    for (int L = 0; L < kv->n_layer; L++) { free(kv->k[L]); free(kv->v[L]); }
    if (kv->px_k) for (int L = 0; L < kv->n_layer; L++) { free(kv->px_k[L]); free(kv->px_v[L]); }
    free(kv->px_k); free(kv->px_v);
    free(kv->k); free(kv->v); free(kv->kv_dim); free(kv->seq); free(kv->f16);
    kv->k = kv->v = NULL; kv->kv_dim = kv->seq = kv->f16 = NULL;
    kv->px_k = kv->px_v = NULL;
}

// ---- the forward pass ---------------------------------------------------------

// Build the per-layer input vectors (PLE): one n_embd_per_layer slice per layer.
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

// The decoder body, fed an already-built input row (see model-cpu.c: token
// embeddings arrive pre-scaled by sqrt(n_embd), media rows as the projector
// made them; ple_token picks the per-layer-input row, -1 for none).
static void forward_core(struct model *m, struct kvcache *kv, const float *x_in, int ple_token,
                         int pos, float *logits) {
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

    float *x   = malloc((size_t)n_embd * sizeof(float));
    float *h   = malloc((size_t)n_embd * sizeof(float));
    float *q   = malloc((size_t)q_max  * sizeof(float));
    float *kb  = malloc((size_t)kv_max * sizeof(float));
    float *vb  = malloc((size_t)kv_max * sizeof(float));
    float *xb  = malloc((size_t)q_max  * sizeof(float));
    float *o   = malloc((size_t)n_embd * sizeof(float));
    float *g1  = malloc((size_t)n_ff   * sizeof(float));
    float *g2  = malloc((size_t)n_ff   * sizeof(float));
    // per-head attention scores: the head loop threads across cores here (the
    // GPU holds the matmuls, so untouched, attention would become the new
    // serial bottleneck as the context grows)
    float *att = malloc((size_t)n_head * (size_t)(pos + 1) * sizeof(float));
    float *pg  = ple > 0 ? malloc((size_t)ple * sizeof(float)) : NULL;

    memcpy(x, x_in, (size_t)n_embd * sizeof(float));

    // per-layer inputs (built from the position's input row)
    float *inp_per_layer = ple_token >= 0 ? build_per_layer(m, ple_token, x) : NULL;

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
            int row = pos % kv->seq[L];                  // ring write (identity on global layers)
            if (kv->f16[L]) {                            // global rows round through f16 once here
                uint16_t *kr = (uint16_t *)kv->k[L] + (size_t)row * kv_dim;
                uint16_t *vr = (uint16_t *)kv->v[L] + (size_t)row * kv_dim;
                for (int i = 0; i < kv_dim; i++) { kr[i] = f16_of(kb[i]); vr[i] = f16_of(vb[i]); }
            } else {
                memcpy((float *)kv->k[L] + (size_t)row * kv_dim, kb, (size_t)kv_dim * sizeof(float));
                memcpy((float *)kv->v[L] + (size_t)row * kv_dim, vb, (size_t)kv_dim * sizeof(float));
            }
        }
        const void *Kc = kv->k[src], *Vc = kv->v[src];
        const int seq = kv->seq[src];                    // ring length of the owning layer
        const int kf16 = kv->f16[src];

        int start = (local && c->sliding_window > 0 && pos - c->sliding_window + 1 > 0)
                  ? pos - c->sliding_window + 1 : 0;

        const int gqa = n_head / n_head_kv;
        int hh;
        #pragma omp parallel for schedule(static)
        for (hh = 0; hh < n_head; hh++) {
            const float *qh = q + hh * hd;
            float *ah = att + (size_t)hh * (pos + 1);
            int kvh = hh / gqa;
            for (int t = start; t <= pos; t++) {
                size_t off = (size_t)(t % seq) * kv_dim + (size_t)kvh * hd;
                float s = 0.0f;
                for (int i = 0; i < hd; i++) s += qh[i] * kv_at(Kc, off + i, kf16);
                ah[t] = s; // Gemma4 attention scale is 1.0 (no 1/sqrt(d))
            }
            softmax(ah + start, pos - start + 1);
            float *outh = xb + hh * hd;
            for (int i = 0; i < hd; i++) outh[i] = 0.0f;
            for (int t = start; t <= pos; t++) {
                size_t off = (size_t)(t % seq) * kv_dim + (size_t)kvh * hd;
                float a = ah[t];
                for (int i = 0; i < hd; i++) outh[i] += a * kv_at(Vc, off + i, kf16);
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

    // final norm + tied output projection (logits = token_embd . x), then softcap.
    // Skipped entirely for prefill (logits == NULL): a prompt token only needs
    // its kv writes, and the head is the largest matmul in the model.
    if (logits) {
        rmsnorm(x, x, fptr(m, "output_norm.weight"), n_embd, eps);
        if (m->last_hidden)                       // h_prev for the MTP draft head
            memcpy(m->last_hidden, x, (size_t)n_embd * sizeof(float));
        matmul_q(logits, wq(m, "token_embd.weight"), x, n_embd, c->n_vocab);
        if (c->logit_softcap > 0.0f) {
            float sc = c->logit_softcap;
            for (int v = 0; v < c->n_vocab; v++) logits[v] = sc * tanhf(logits[v] / sc);
        }
    }

    free(x); free(h); free(q); free(kb); free(vb); free(xb);
    free(o); free(g1); free(g2); free(att); free(pg);
    free(inp_per_layer);   // rope_freqs and all weights are owned by the cache
}

void model_forward(struct model *m, struct kvcache *kv, int token, int pos, float *logits) {
    // embedding lookup (one quantized row), scaled by sqrt(n_embd)
    const int n_embd = m->cfg.n_embd;
    float *erow = dequantize_row(wq(m, "token_embd.weight"), token, n_embd);
    for (int i = 0; i < n_embd; i++) erow[i] *= sqrtf((float)n_embd);
    forward_core(m, kv, erow, token, pos, logits);
    free(erow);
}

void model_prefill_embd(struct model *m, struct kvcache *kv, const float *rows, int n, int pos0) {
    // On PLE models a media position takes the PADDING token's (id 0)
    // per-layer row beside the usual projection of its embedding.
    for (int i = 0; i < n; i++)
        forward_core(m, kv, rows + (size_t)i * m->cfg.n_embd, 0, pos0 + i, NULL);
}

// Forward + greedy pick. The logits still cross back from the GPU each token
// (a device argmax would send 4 bytes instead — a measured next step, not v1).
int model_forward_next(struct model *m, struct kvcache *kv, int token, int pos) {
    static float *lbuf = NULL;
    if (!lbuf) {
        lbuf = malloc((size_t)m->cfg.n_vocab * sizeof(float));
        if (!lbuf) return -1;
    }
    model_forward(m, kv, token, pos, lbuf);
    int best = 0;
    for (int v = 1; v < m->cfg.n_vocab; v++) if (lbuf[v] > lbuf[best]) best = v;
    return best;
}

void model_prefill(struct model *m, struct kvcache *kv, const int *tokens, int n, int pos0) {
    // No chunking yet: each prompt token is one forward that skips the head,
    // like the CPU backend. Chunked prefill (each weight matrix crossing DRAM
    // once per chunk instead of once per token) is THE known next step here —
    // it is most of the CUDA backends' prefill speed.
    for (int i = 0; i < n; i++) model_forward(m, kv, tokens[i], pos0 + i, NULL);
}

void model_prefill_mixed(struct model *m, struct kvcache *kv, const float *rows,
                         const int *ids, int n, int pos0) {
    // "Mixed" is just dispatch here: the packing this API exists for is a
    // chunk-boundary story, and this backend has no chunks yet.
    for (int i = 0; i < n; i++) {
        if (ids[i] >= 0) model_forward(m, kv, ids[i], pos0 + i, NULL);
        else forward_core(m, kv, rows + (size_t)(-ids[i] - 1) * m->cfg.n_embd, 0, pos0 + i, NULL);
    }
}
