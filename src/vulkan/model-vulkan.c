// Vulkan backend: model-cpu.c's readable forward with the quantized matmul —
// nearly all of a token's weight traffic and compute — dispatched to the GPU
// (vk-compute.c, one compute pipeline per weight type). Everything else
// (norms, RoPE, attention over the host KV cache, GELU) stays the CPU
// reference code: on the integrated GPUs this backend targets, device memory
// IS system RAM, so the activation handoff each way is a memcpy, not a bus.
//
// The forward is written over a CHUNK of B positions (decode: B=1). Prefill
// is bandwidth-bound — each token's forward streams every weight matrix
// through DRAM — so the chunk form lets one weight pass serve B tokens (the
// wide NB=16 shader), which is most of prefill's cost; and it groups the
// independent matmuls (q/k/v, gate/up) into single submits, one fence
// round-trip instead of three. Each wide column accumulates in the same
// element order as the narrow kernel, so chunked prefill is bit-identical to
// token-at-a-time — same words out, only the clock moves.
//
// LG_VK_VERIFY=1 runs every GPU matmul beside the host oracle and reports the
// max difference (the LG_MEDIA_VERIFY pattern): expected ~1e-4-scale, pure
// summation-order reassociation — the dequantized values are bit-identical.
// LG_VK_PREFILL_B=n clamps the prefill chunk width (1..16).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "model.h"
#include "quant.h"
#include "vulkan/vk-compute.h"

#define CHUNK_MAX VKC_NB_MAX      // prefill chunk width; SWA rings pad by this

const int model_kv_host = 1;     // this backend's kvcache rows are host memory

static int chunk_width(void) {
    static int b = 0;
    if (!b) {
        const char *e = getenv("LG_VK_PREFILL_B");
        b = e ? atoi(e) : CHUNK_MAX;
        if (b < 1) b = 1;
        if (b > CHUNK_MAX) b = CHUNK_MAX;
    }
    return b;
}

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

// model_forward_spec lives below forward_chunk: the MTP verify runs as ONE
// batched chunk — the weights cross memory once for the whole block, which
// is what makes an accepted draft nearly free.

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

// njobs matmuls over one shared x of nb columns: GPU as one submit, host
// column-by-column when a weight type has no shader (announced once per
// type, never silently).
static void matmul_jobs(struct vkc_job *jobs, int njobs, const float *x, int k, int nb) {
    static int verify = -1;
    static uint32_t warned = 0;               // one bit per ggml type id
    if (verify < 0) verify = getenv("LG_VK_VERIFY") != NULL;

    if (vkc_matmul_x(x, k, nb, jobs, njobs) != 0) {
        for (int j = 0; j < njobs; j++) {
            uint32_t ty = jobs[j].t->type;
            if (!vkc_supported(ty) && ty < 32 && !(warned & (1u << ty))) {
                warned |= 1u << ty;
                fprintf(stderr, "vulkan: no %s pipeline — %s stays on the host matmul\n",
                        ggml_type_name(ty), jobs[j].t->name);
            }
            for (int b = 0; b < nb; b++)
                matmul_host(jobs[j].out + (size_t)b * jobs[j].m, jobs[j].t,
                            x + (size_t)b * k, k, jobs[j].m);
        }
        return;
    }
    if (verify) {
        static float worst = 0.0f;
        static int calls = 0;
        for (int j = 0; j < njobs; j++) {
            float *ref = malloc((size_t)jobs[j].m * sizeof(float));
            if (!ref) return;
            float md = 0.0f, scale = 0.0f;
            for (int b = 0; b < nb; b++) {
                matmul_host(ref, jobs[j].t, x + (size_t)b * k, k, jobs[j].m);
                const float *got = jobs[j].out + (size_t)b * jobs[j].m;
                for (int i = 0; i < jobs[j].m; i++) {
                    float d = fabsf(got[i] - ref[i]);
                    if (d > md) md = d;
                    float a = fabsf(ref[i]);
                    if (a > scale) scale = a;
                }
            }
            calls++;
            if (calls <= 16 || md > worst)
                fprintf(stderr, "vulkan verify[%d]: %-6s %6d x %-6d nb=%-2d max|gpu-host| %.3e  (max|host| %.3e)\n",
                        calls, ggml_type_name(jobs[j].t->type), k, jobs[j].m, nb, md, scale);
            if (md > worst) worst = md;
            free(ref);
        }
    }
}

// out[m] = W . x, the single-matmul shape.
static void matmul_q(float *out, const struct gguf_tensor *t, const float *x, int k, int m) {
    struct vkc_job job = { .t = t, .out = out, .m = m };
    matmul_jobs(&job, 1, x, k, 1);
}

// B-column form: out[b*m + row] = W . x_b.
static void matmul_qb(float *out, const struct gguf_tensor *t, const float *x, int k, int m, int B) {
    struct vkc_job job = { .t = t, .out = out, .m = m };
    matmul_jobs(&job, 1, x, k, B);
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
        // Sliding-window layers keep a ring of window rows, PADDED by the
        // prefill chunk width: a chunk writes all B of its rows before any of
        // its attention runs, and with an exact-window ring row (p % window)
        // for the chunk's last token would land on a position the chunk's
        // FIRST token still attends. window+B rows make every clobbered row
        // one that no token of the chunk can reach — the same padding the
        // CUDA cache applies for its prefill chunks (see model-cpu.c).
        int seq = max_seq;
        if (m->is_local[L] && c->sliding_window > 0 && c->sliding_window + CHUNK_MAX < max_seq)
            seq = c->sliding_window + CHUNK_MAX;
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

// Per-layer input vectors (PLE) for a whole chunk: token b's [n_layer][ple]
// block at ret + b*total. One batched projection instead of B — the 55 MB
// per_layer_model_proj crosses DRAM once per chunk.
static float *build_per_layer_chunk(struct model *m, const int *toks,
                                    const float *x /*[B][n_embd]*/, int B) {
    const struct config *c = &m->cfg;
    const int ple = c->n_embd_per_layer;
    if (ple <= 0) return NULL;
    const struct gguf_tensor *pte = gguf_find_tensor(m->ctx, "per_layer_token_embd.weight");
    if (!pte) return NULL;
    const int64_t total = (int64_t)ple * c->n_layer;

    float *proj = malloc((size_t)B * total * sizeof(float));
    if (!proj) return NULL;
    matmul_qb(proj, wq(m, "per_layer_model_proj.weight"), x, c->n_embd, (int)total, B);

    const float *pn = fptr(m, "per_layer_proj_norm.weight");
    const float pscale = 1.0f / sqrtf((float)c->n_embd);
    const float te_scale = sqrtf((float)ple);
    const float inv_sqrt2 = 1.0f / sqrtf(2.0f);
    for (int b = 0; b < B; b++) {
        float *pb = proj + (size_t)b * total;
        for (int64_t i = 0; i < total; i++) pb[i] *= pscale;
        for (int L = 0; L < c->n_layer; L++)
            rmsnorm(pb + (size_t)L * ple, pb + (size_t)L * ple, pn, ple, c->rms_eps);
        // token b's per-layer embedding row, scaled by sqrt(ple)
        float *tok = dequantize_row(pte, toks[b], total);   // per token, not cached
        if (!tok) { free(proj); return NULL; }
        for (int64_t i = 0; i < total; i++) pb[i] = (pb[i] + tok[i] * te_scale) * inv_sqrt2;
        free(tok);
    }
    return proj;
}

// The decoder body over a CHUNK of B consecutive positions pos0..pos0+B-1.
// x_in is [B][n_embd], each row exactly as the layers should see it (token
// embeddings pre-scaled by sqrt(n_embd), media rows as the projector made
// them). ple_tok[b] picks position b's per-layer-input row (media positions
// pass the padding token, 0 — the reference does the same for embedding
// batches). When `logits` is non-NULL the head runs for the last `nlogits`
// positions (decode: 1; MTP verify: the whole block) — during prefill nobody
// reads any, and the head is the largest matmul in the model. hidden_out, if
// given, receives those positions' post-norm hiddens ([nlogits][n_embd]).
static void forward_chunk(struct model *m, struct kvcache *kv, const float *x_in,
                          const int *ple_tok, int pos0, int B,
                          float *logits, int nlogits, float *hidden_out) {
    const struct config *c = &m->cfg;
    const int n_embd = c->n_embd, n_head = c->n_head;
    const int n_ff = c->n_ff, ple = c->n_embd_per_layer;
    const float eps = c->rms_eps;
    const int pos_last = pos0 + B - 1;

    // scratch sized for the widest layer (head_dim and n_head_kv vary per layer)
    int maxhd = 0, max_kvdim = 0;
    for (int L = 0; L < c->n_layer; L++) {
        if (m->head_dim[L] > maxhd) maxhd = m->head_dim[L];
        int kvd = m->n_head_kv[L] * m->head_dim[L];
        if (kvd > max_kvdim) max_kvdim = kvd;
    }
    const int q_max  = n_head * maxhd;
    const int kv_max = max_kvdim;

    // per-chunk scratch: [B][len], contiguous so a whole matrix uploads as one
    // memcpy and reads back the same way
    float *x   = malloc((size_t)B * n_embd * sizeof(float));
    float *h   = malloc((size_t)B * n_embd * sizeof(float));
    float *q   = malloc((size_t)B * q_max  * sizeof(float));
    float *kb  = malloc((size_t)B * kv_max * sizeof(float));
    float *vb  = malloc((size_t)B * kv_max * sizeof(float));
    float *xb  = malloc((size_t)B * q_max  * sizeof(float));
    float *o   = malloc((size_t)B * n_embd * sizeof(float));
    float *g1  = malloc((size_t)B * n_ff   * sizeof(float));
    float *g2  = malloc((size_t)B * n_ff   * sizeof(float));
    // per-head attention scores (reused per token): threads across cores — the
    // GPU holds the matmuls, so untouched, attention would become the serial
    // bottleneck as the context grows
    float *att = malloc((size_t)n_head * (size_t)(pos_last + 1) * sizeof(float));
    float *pg  = ple > 0 ? malloc((size_t)B * ple * sizeof(float)) : NULL;
    // f32 staging for the f16-stored global-layer KV rows: scalar f16->f32 per
    // element inside the attention loops dominated long-context time (every
    // q-head of a GQA group re-converted its kv-head's rows, every token of
    // the chunk again) — converting each row ONCE per layer per chunk makes
    // attention pure f32 math over shared staging
    float *kf32 = malloc((size_t)(pos_last + 1) * kv_max * sizeof(float));
    float *vf32 = malloc((size_t)(pos_last + 1) * kv_max * sizeof(float));

    memcpy(x, x_in, (size_t)B * n_embd * sizeof(float));

    // per-layer inputs for the whole chunk (from each position's input row)
    float *inp_per_layer = build_per_layer_chunk(m, ple_tok, x, B);
    const int64_t ple_total = (int64_t)ple * c->n_layer;

    // rope_freqs (freq_factors, stored f32) used by global/full layers only
    const float *rope_freqs = fptr(m, "rope_freqs.weight");

    int staged_src = -1;   // which layer's f16 rows currently sit in kf32/vf32

    for (int L = 0; L < c->n_layer; L++) {
        const int local = m->is_local[L];
        const int hd = head_dim_at(m, L);
        const int n_head_kv = m->n_head_kv[L];
        const int q_dim = n_head * hd, kv_dim = n_head_kv * hd;
        const float base = local ? c->rope_freq_base_swa : c->rope_freq_base;
        const float *ff = local ? NULL : rope_freqs;

        // ---- attention ----
        for (int b = 0; b < B; b++)
            rmsnorm(h + (size_t)b * n_embd, x + (size_t)b * n_embd,
                    fptr_layer(m, L, "attn_norm.weight"), n_embd, eps);

        // q, k, v: independent matmuls over the same h — ONE submit. Layers
        // past n_kv_start reuse an earlier ring and have no k/v projections.
        const struct gguf_tensor *wv = wq_layer(m, L, "attn_v.weight");
        struct vkc_job jobs[3] = {
            { .t = wq_layer(m, L, "attn_q.weight"), .out = q,  .m = q_dim  },
            { .t = wq_layer(m, L, "attn_k.weight"), .out = kb, .m = kv_dim },
            { .t = wv,                              .out = vb, .m = kv_dim },
        };
        int own = L < c->n_kv_start;
        matmul_jobs(jobs, own ? (wv ? 3 : 2) : 1, h, n_embd, B);

        const float *qn = fptr_layer(m, L, "attn_q_norm.weight");
        for (int b = 0; b < B; b++) {
            float *qb_ = q + (size_t)b * q_dim;
            for (int hh = 0; hh < n_head; hh++) rmsnorm(qb_ + hh * hd, qb_ + hh * hd, qn, hd, eps);
            for (int hh = 0; hh < n_head; hh++) rope_neox(qb_ + hh * hd, hd, pos0 + b, base, ff);
        }

        int src = kv_src(m, L);
        if (own) {                               // this layer owns its KV
            const float *kn = fptr_layer(m, L, "attn_k_norm.weight");
            for (int b = 0; b < B; b++) {
                float *kbb = kb + (size_t)b * kv_dim;
                float *vbb = vb + (size_t)b * kv_dim;
                if (!wv) memcpy(vbb, kbb, (size_t)kv_dim * sizeof(float)); // no V proj: V = K projection
                for (int hh = 0; hh < n_head_kv; hh++) rmsnorm(kbb + hh * hd, kbb + hh * hd, kn, hd, eps);
                for (int hh = 0; hh < n_head_kv; hh++) rmsnorm_plain(vbb + hh * hd, vbb + hh * hd, hd, eps);
                for (int hh = 0; hh < n_head_kv; hh++) rope_neox(kbb + hh * hd, hd, pos0 + b, base, ff);
                int row = (pos0 + b) % kv->seq[L];           // ring write (identity on global layers)
                if (kv->f16[L]) {                            // global rows round through f16 once here
                    uint16_t *kr = (uint16_t *)kv->k[L] + (size_t)row * kv_dim;
                    uint16_t *vr = (uint16_t *)kv->v[L] + (size_t)row * kv_dim;
                    for (int i = 0; i < kv_dim; i++) { kr[i] = f16_of(kbb[i]); vr[i] = f16_of(vbb[i]); }
                } else {
                    memcpy((float *)kv->k[L] + (size_t)row * kv_dim, kbb, (size_t)kv_dim * sizeof(float));
                    memcpy((float *)kv->v[L] + (size_t)row * kv_dim, vbb, (size_t)kv_dim * sizeof(float));
                }
            }
        }
        const int seq = kv->seq[src];                    // ring length of the owning layer
        const int kf16 = kv->f16[src];
        const int gqa = n_head / n_head_kv;

        // f16 layers: stage rows 0..pos_last as f32 once, indexed by POSITION
        // (globals never wrap, but go through the ring map anyway); f32 rings
        // are read in place. Every token and every head then does f32 math.
        const float *Kf, *Vf;
        int by_pos = kf16;
        if (kf16) {
            if (src != staged_src) {   // reusing layers share one source: stage once
                const uint16_t *K16 = kv->k[src], *V16 = kv->v[src];
                int t;
                #pragma omp parallel for schedule(static)
                for (t = 0; t <= pos_last; t++) {
                    size_t src_off = (size_t)(t % seq) * kv_dim;
                    size_t dst_off = (size_t)t * kv_dim;
                    for (int i = 0; i < kv_dim; i++) {
                        kf32[dst_off + i] = f16_to_f32(K16[src_off + i]);
                        vf32[dst_off + i] = f16_to_f32(V16[src_off + i]);
                    }
                }
                staged_src = src;
            }
            Kf = kf32; Vf = vf32;
        } else {
            Kf = kv->k[src]; Vf = kv->v[src];
        }

        // all B rows of this layer's KV are seated above; token b attends
        // positions <= pos0+b, all of which now exist — causality needs only
        // completed prefixes, exactly why chunked prefill is legal at all
        for (int b = 0; b < B; b++) {
            const int pos = pos0 + b;
            int start = (local && c->sliding_window > 0 && pos - c->sliding_window + 1 > 0)
                      ? pos - c->sliding_window + 1 : 0;
            float *qb_ = q + (size_t)b * q_dim;
            float *xbb = xb + (size_t)b * q_dim;
            int hh;
            #pragma omp parallel for schedule(static)
            for (hh = 0; hh < n_head; hh++) {
                const float *qh = qb_ + hh * hd;
                float *ah = att + (size_t)hh * (pos + 1);
                int kvh = hh / gqa;
                for (int t = start; t <= pos; t++) {
                    size_t off = (size_t)(by_pos ? t : t % seq) * kv_dim + (size_t)kvh * hd;
                    const float *kr = Kf + off;
                    float s = 0.0f;
                    for (int i = 0; i < hd; i++) s += qh[i] * kr[i];
                    ah[t] = s; // Gemma4 attention scale is 1.0 (no 1/sqrt(d))
                }
                softmax(ah + start, pos - start + 1);
                float *outh = xbb + hh * hd;
                for (int i = 0; i < hd; i++) outh[i] = 0.0f;
                for (int t = start; t <= pos; t++) {
                    size_t off = (size_t)(by_pos ? t : t % seq) * kv_dim + (size_t)kvh * hd;
                    const float *vr = Vf + off;
                    float a = ah[t];
                    for (int i = 0; i < hd; i++) outh[i] += a * vr[i];
                }
            }
        }

        matmul_qb(o, wq_layer(m, L, "attn_output.weight"), xb, q_dim, n_embd, B);
        for (int b = 0; b < B; b++)
            rmsnorm(o + (size_t)b * n_embd, o + (size_t)b * n_embd,
                    fptr_layer(m, L, "post_attention_norm.weight"), n_embd, eps);
        for (size_t i = 0; i < (size_t)B * n_embd; i++) x[i] += o[i];   // attn residual

        // ---- feed-forward (GeGLU); width is per-layer (elastic FFN) ----
        const int nff = m->ffn_len[L];
        for (int b = 0; b < B; b++)
            rmsnorm(h + (size_t)b * n_embd, x + (size_t)b * n_embd,
                    fptr_layer(m, L, "ffn_norm.weight"), n_embd, eps);
        // gate and up share h: one submit. Columns pack at stride nff (the
        // LAYER'S width) so the down matmul's input columns are contiguous.
        struct vkc_job fjobs[2] = {
            { .t = wq_layer(m, L, "ffn_gate.weight"), .out = g1, .m = nff },
            { .t = wq_layer(m, L, "ffn_up.weight"),   .out = g2, .m = nff },
        };
        matmul_jobs(fjobs, 2, h, n_embd, B);
        {
            long i;
            #pragma omp parallel for schedule(static)
            for (i = 0; i < (long)B * nff; i++) g1[i] = gelu(g1[i]) * g2[i];
        }
        matmul_qb(o, wq_layer(m, L, "ffn_down.weight"), g1, nff, n_embd, B);
        for (int b = 0; b < B; b++)
            rmsnorm(o + (size_t)b * n_embd, o + (size_t)b * n_embd,
                    fptr_layer(m, L, "post_ffw_norm.weight"), n_embd, eps);
        for (size_t i = 0; i < (size_t)B * n_embd; i++) x[i] += o[i];   // ffn residual

        // ---- per-layer input (PLE) ----
        if (inp_per_layer) {
            matmul_qb(pg, wq_layer(m, L, "inp_gate.weight"), x, n_embd, ple, B);
            for (int b = 0; b < B; b++) {
                const float *ile = inp_per_layer + (size_t)b * ple_total + (size_t)L * ple;
                float *pgb = pg + (size_t)b * ple;
                for (int i = 0; i < ple; i++) pgb[i] = gelu(pgb[i]) * ile[i];
            }
            matmul_qb(o, wq_layer(m, L, "proj.weight"), pg, ple, n_embd, B);
            for (int b = 0; b < B; b++)
                rmsnorm(o + (size_t)b * n_embd, o + (size_t)b * n_embd,
                        fptr_layer(m, L, "post_norm.weight"), n_embd, eps);
            for (size_t i = 0; i < (size_t)B * n_embd; i++) x[i] += o[i];   // PLE residual
        }

        // ---- per-layer output scale (f32 scalar) ----
        const float *os = fptr_layer(m, L, "layer_output_scale.weight");
        if (os) { for (size_t i = 0; i < (size_t)B * n_embd; i++) x[i] *= os[0]; }
    }

    // final norm + tied output projection (logits = token_embd . x), then
    // softcap — for the chunk's last nlogits positions only; prefill
    // positions have no reader and the head is the largest matmul in the model.
    if (logits && nlogits > 0) {
        float *xt0 = x + (size_t)(B - nlogits) * n_embd;
        for (int t = 0; t < nlogits; t++)
            rmsnorm(xt0 + (size_t)t * n_embd, xt0 + (size_t)t * n_embd,
                    fptr(m, "output_norm.weight"), n_embd, eps);
        if (hidden_out)
            memcpy(hidden_out, xt0, (size_t)nlogits * n_embd * sizeof(float));
        if (m->last_hidden)                       // h_prev for the MTP draft head
            memcpy(m->last_hidden, x + (size_t)(B - 1) * n_embd, (size_t)n_embd * sizeof(float));
        matmul_qb(logits, wq(m, "token_embd.weight"), xt0, n_embd, c->n_vocab, nlogits);
        if (c->logit_softcap > 0.0f) {
            float sc = c->logit_softcap;
            for (size_t v = 0; v < (size_t)nlogits * c->n_vocab; v++)
                logits[v] = sc * tanhf(logits[v] / sc);
        }
    }

    free(x); free(h); free(q); free(kb); free(vb); free(xb);
    free(o); free(g1); free(g2); free(att); free(pg); free(kf32); free(vf32);
    free(inp_per_layer);   // rope_freqs and all weights are owned by the cache
}

void model_forward(struct model *m, struct kvcache *kv, int token, int pos, float *logits) {
    // embedding lookup (one quantized row), scaled by sqrt(n_embd)
    const int n_embd = m->cfg.n_embd;
    float *erow = dequantize_row(wq(m, "token_embd.weight"), token, n_embd);
    for (int i = 0; i < n_embd; i++) erow[i] *= sqrtf((float)n_embd);
    forward_chunk(m, kv, erow, &token, pos, 1, logits, 1, NULL);
    free(erow);
}

// The MTP verify as one batched chunk: LG_MTP_N tokens forward together, the
// head runs over all of them, and greedy argmax walks the block. Byte-
// identical to sequential verification by construction — each wide column
// accumulates in the narrow kernel's element order, and out[j] is only read
// while every earlier draft held, exactly the positions whose cache rows
// were seated with the correct tokens (rows past the first miss hold draft
// garbage until the real tokens rewrite them, and nothing reads them first).
int model_forward_spec(struct model *m, struct kvcache *kv, const int *toks, int pos, int *out) {
    const int B = LG_MTP_N, n_embd = m->cfg.n_embd, n_vocab = m->cfg.n_vocab;
    static float *logits = NULL, *hidden = NULL, *rows = NULL;
    if (!logits) {
        logits = malloc((size_t)B * n_vocab * sizeof(float));
        hidden = malloc((size_t)B * n_embd * sizeof(float));
        rows   = malloc((size_t)B * n_embd * sizeof(float));
        if (!logits || !hidden || !rows) return -1;
    }
    const float esc = sqrtf((float)n_embd);
    for (int b = 0; b < B; b++) {
        float *erow = dequantize_row(wq(m, "token_embd.weight"), toks[b], n_embd);
        if (!erow) return -1;
        for (int i = 0; i < n_embd; i++) rows[(size_t)b * n_embd + i] = erow[i] * esc;
        free(erow);
    }
    forward_chunk(m, kv, rows, toks, pos, B, logits, B, hidden);

    int j = 0;
    do {
        const float *lb = logits + (size_t)j * n_vocab;
        int best = 0;
        for (int v = 1; v < n_vocab; v++) if (lb[v] > lb[best]) best = v;
        out[j] = best;
        j++;
    } while (j < B && out[j - 1] == toks[j]);
    // h_prev for the next draft: the last VALID position's hidden
    memcpy(m->last_hidden, hidden + (size_t)(j - 1) * n_embd, (size_t)n_embd * sizeof(float));
    return j;
}

void model_prefill_embd(struct model *m, struct kvcache *kv, const float *rows, int n, int pos0) {
    // On PLE models a media position takes the PADDING token's (id 0)
    // per-layer row beside the usual projection of its embedding.
    static const int zeros[CHUNK_MAX];
    const int W = chunk_width();
    for (int i = 0; i < n; i += W) {
        int B = n - i < W ? n - i : W;
        forward_chunk(m, kv, rows + (size_t)i * m->cfg.n_embd, zeros, pos0 + i, B, NULL, 0, NULL);
    }
}

// Forward + greedy pick. The logits still cross back from the GPU each token
// (a device argmax would send 4 bytes instead — a measured next step, not now).
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
    // Chunked: W tokens share each weight pass through the wide pipeline —
    // prefill is bandwidth-bound, so that factor is most of its cost. The
    // outputs are bit-identical to token-at-a-time (same kernel, same order
    // per column); only the clock moves.
    const int n_embd = m->cfg.n_embd;
    const int W = chunk_width();
    const float esc = sqrtf((float)n_embd);
    float *rows = malloc((size_t)W * n_embd * sizeof(float));
    if (!rows) return;
    for (int i = 0; i < n; i += W) {
        int B = n - i < W ? n - i : W;
        for (int b = 0; b < B; b++) {
            float *erow = dequantize_row(wq(m, "token_embd.weight"), tokens[i + b], n_embd);
            if (!erow) { free(rows); return; }
            for (int j = 0; j < n_embd; j++) rows[(size_t)b * n_embd + j] = erow[j] * esc;
            free(erow);
        }
        forward_chunk(m, kv, rows, tokens + i, pos0 + i, B, NULL, 0, NULL);
    }
    free(rows);
}

void model_prefill_mixed(struct model *m, struct kvcache *kv, const float *rows,
                         const int *ids, int n, int pos0) {
    // One stream, chunked without regard to the text/media seams — that is
    // the point of this API (a turn prefilled as separate segments pays a
    // weight pass per segment; see model.h).
    const int n_embd = m->cfg.n_embd;
    const int W = chunk_width();
    const float esc = sqrtf((float)n_embd);
    float *xin = malloc((size_t)W * n_embd * sizeof(float));
    int ple_tok[CHUNK_MAX];
    if (!xin) return;
    for (int i = 0; i < n; i += W) {
        int B = n - i < W ? n - i : W;
        for (int b = 0; b < B; b++) {
            int id = ids[i + b];
            if (id >= 0) {                       // text token: embed + scale
                float *erow = dequantize_row(wq(m, "token_embd.weight"), id, n_embd);
                if (!erow) { free(xin); return; }
                for (int j = 0; j < n_embd; j++) xin[(size_t)b * n_embd + j] = erow[j] * esc;
                free(erow);
                ple_tok[b] = id;
            } else {                             // media row: entered as given
                memcpy(xin + (size_t)b * n_embd,
                       rows + (size_t)(-id - 1) * n_embd, (size_t)n_embd * sizeof(float));
                ple_tok[b] = 0;
            }
        }
        forward_chunk(m, kv, xin, ple_tok, pos0 + i, B, NULL, 0, NULL);
    }
    free(xin);
}
