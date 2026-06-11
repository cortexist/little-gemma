// MTP — the gemma4-assistant draft head (multi-token prediction).
//
// A tiny transformer (E2B's: 4 blocks, 256 wide, 77M params, mostly its own
// LM head) that predicts the token AFTER next. It owns no K or V projections
// at all: every block cross-attends straight into the TARGET model's KV cache
// — sliding-window blocks read the target's last SWA KV layer, the full block
// reads the last global KV layer. Its inputs are the target's own embedding
// of the last chosen token concatenated with the target's last hidden state,
// squeezed through the next-token pre-projection; the post-projection lifts
// the result back to target width so steps can chain. (Today's files name
// these tensors "nextn.*"; the loader also accepts the more literal
// "next_token.*" for when the converter is renamed.)
//
// Drafts are verified by the target (greedy match), so MTP NEVER changes the
// output — only how many target forwards run per emitted token. That is the
// correctness gate: same binary with and without -mtp must emit identical
// text. The win is hardware-dependent: batching/overlap converts ~nothing on
// the latency-bound A5000 but +40% on the bandwidth-bound Orin (user's fork
// measurements); spec blocks beyond 2 regressed there, so this drafts ONE
// token per round.
//
// Host f32 implementation in the model-cpu.c idiom; it reads struct kvcache
// rows directly, so it requires a backend whose cache lives in host memory
// (model_kv_host — the CPU backend; the CUDA ports keep their own copy).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "gguf.h"
#include "quant.h"
#include "model.h"
#include "mtp-internal.h"

// ---- the same small math model-cpu.c uses -----------------------------------

static void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float s = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * s * (w ? w[i] : 1.0f);
}

static void rope_neox(float *v, int d, int pos, float base) {
    int half = d / 2;
    for (int i = 0; i < half; i++) {
        float freq = powf(base, -2.0f * (float)i / (float)d);
        float ang = (float)pos * freq;
        float c = cosf(ang), s = sinf(ang);
        float a = v[i], b = v[i + half];
        v[i]        = a * c - b * s;
        v[i + half] = a * s + b * c;
    }
}

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

static void matmul_q(float *out, const struct gguf_tensor *t, const float *x, int k, int m) {
    const int blck = ggml_blck_size(t->type);
    const size_t row_bytes = (size_t)(k / blck) * ggml_type_size(t->type);
    const unsigned char *base = t->data;
    #pragma omp parallel
    {
        float *buf = malloc((size_t)k * sizeof(float));
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

// f16 read for the target's global-layer KV rows (bit-matches model-cpu.c).
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t em   = h & 0x7fffu;
    uint32_t r;
    if (em >= 0x7c00u)      r = sign | 0x7f800000u | ((em & 0x3ffu) << 13);   // inf/nan
    else if (em >= 0x0400u) r = sign | ((em + 0x1c000u) << 13);               // normal
    else if (em == 0)       r = sign;                                          // zero
    else {                                                                     // subnormal
        int e = -1; uint32_t mant = em;
        while (!(mant & 0x0400u)) { mant <<= 1; e--; }
        r = sign | ((uint32_t)(127 - 15 + e + 1) << 23) | ((mant & 0x3ffu) << 13);
    }
    float f; memcpy(&f, &r, 4);
    return f;
}
static float kv_at(const void *row, size_t i, int f16) {
    return f16 ? f16_to_f32(((const uint16_t *)row)[i]) : ((const float *)row)[i];
}

// ---- open / free -------------------------------------------------------------

static const float *fptr_l(struct mtp *t, int L, const char *suffix) {
    char name[96];
    snprintf(name, sizeof name, "blk.%d.%s", L, suffix);
    const struct gguf_tensor *w = gguf_find_tensor(t->ctx, name);
    return w ? (const float *)w->data : NULL;
}
static const struct gguf_tensor *wq_l(struct mtp *t, int L, const char *suffix) {
    char name[96];
    snprintf(name, sizeof name, "blk.%d.%s", L, suffix);
    return gguf_find_tensor(t->ctx, name);
}

struct mtp *mtp_open(const char *path, const struct model *m) {
    struct gguf_context *ctx = load_gguf(path);
    if (!ctx) return NULL;
    const char *arch = gguf_get_str(ctx, "general.architecture", "");
    if (strcmp(arch, "gemma4-assistant") != 0) {
        fprintf(stderr, "mtp: %s is a '%s' gguf, want gemma4-assistant\n", path, arch);
        free_gguf(ctx);
        return NULL;
    }

    struct mtp *t = calloc(1, sizeof *t);
    if (!t) { free_gguf(ctx); return NULL; }
    t->ctx = ctx;
    t->n_layer  = (int)gguf_get_u32(ctx, "gemma4-assistant.block_count", 0);
    t->n_inner  = (int)gguf_get_u32(ctx, "gemma4-assistant.embedding_length", 0);
    t->n_bb     = (int)gguf_get_u32(ctx, "gemma4-assistant.embedding_length_out", 0);
    t->n_head   = (int)gguf_get_u32(ctx, "gemma4-assistant.attention.head_count", 0);
    t->n_ff     = (int)gguf_get_u32(ctx, "gemma4-assistant.feed_forward_length", 0);
    t->eps      = gguf_get_f32(ctx, "gemma4-assistant.attention.layer_norm_rms_epsilon", 1e-6f);
    t->base_full= gguf_get_f32(ctx, "gemma4-assistant.rope.freq_base", 1e6f);
    t->base_swa = gguf_get_f32(ctx, "gemma4-assistant.rope.freq_base_swa", 1e4f);
    t->softcap  = gguf_get_f32(ctx, "gemma4-assistant.final_logit_softcapping", 0.0f);
    // the next-token projections — "nextn." in today's files, "next_token."
    // once the converter says what it means
    t->pre  = gguf_find_tensor(ctx, "next_token.pre_projection.weight");
    t->post = gguf_find_tensor(ctx, "next_token.post_projection.weight");
    if (!t->pre)  t->pre  = gguf_find_tensor(ctx, "nextn.pre_projection.weight");
    if (!t->post) t->post = gguf_find_tensor(ctx, "nextn.post_projection.weight");
    t->head = gguf_find_tensor(ctx, "token_embd.weight");
    const struct gguf_tensor *on = gguf_find_tensor(ctx, "output_norm.weight");
    t->out_norm = on ? (const float *)on->data : NULL;
    if (!t->n_layer || !t->n_inner || !t->pre || !t->post || !t->head || !t->out_norm ||
        t->n_bb != m->cfg.n_embd || (int)t->pre->dims[0] != 2 * t->n_bb) {
        fprintf(stderr, "mtp: %s does not fit this target (backbone %d vs n_embd %d)\n",
                path, t->n_bb, m->cfg.n_embd);
        mtp_free(t);
        return NULL;
    }
    t->n_vocab = (int)t->head->dims[1];

    // The target KV the blocks attend: the LAST target layer of each attention
    // type — which, past n_kv_start, stores in the last OWNING layer of that
    // type (full: n_kv_start-1, SWA: n_kv_start-2; see model-cpu.c kv_src).
    const struct config *c = &m->cfg;
    int src_full = c->n_kv_start - 1, src_swa = c->n_kv_start - 2;

    t->l = calloc((size_t)t->n_layer, sizeof *t->l);
    if (!t->l) { mtp_free(t); return NULL; }
    for (int L = 0; L < t->n_layer; L++) {
        struct mtp_layer *bl = &t->l[L];
        bl->attn_norm = fptr_l(t, L, "attn_norm.weight");
        bl->q_norm    = fptr_l(t, L, "attn_q_norm.weight");
        bl->post_attn = fptr_l(t, L, "post_attention_norm.weight");
        bl->ffn_norm  = fptr_l(t, L, "ffn_norm.weight");
        bl->post_ffw  = fptr_l(t, L, "post_ffw_norm.weight");
        bl->out_scale = fptr_l(t, L, "layer_output_scale.weight");
        bl->q    = wq_l(t, L, "attn_q.weight");
        bl->o    = wq_l(t, L, "attn_output.weight");
        bl->gate = wq_l(t, L, "ffn_gate.weight");
        bl->up   = wq_l(t, L, "ffn_up.weight");
        bl->down = wq_l(t, L, "ffn_down.weight");
        if (!bl->attn_norm || !bl->q_norm || !bl->post_attn || !bl->ffn_norm ||
            !bl->post_ffw || !bl->q || !bl->o || !bl->gate || !bl->up || !bl->down) {
            fprintf(stderr, "mtp: block %d tensors missing\n", L);
            mtp_free(t);
            return NULL;
        }
        // head dim from the q projection; whether the block is local follows
        // from which target layer has heads that size (no bool-array parsing)
        bl->hd = (int)bl->q->dims[1] / t->n_head;
        bl->local = bl->hd == m->head_dim[src_swa];
        bl->src = bl->local ? src_swa : src_full;
        if (bl->hd != m->head_dim[bl->src]) {
            fprintf(stderr, "mtp: block %d head dim %d matches no target KV layer\n", L, bl->hd);
            mtp_free(t);
            return NULL;
        }
    }

    int q_max = 0;
    for (int L = 0; L < t->n_layer; L++)
        if (t->n_head * t->l[L].hd > q_max) q_max = t->n_head * t->l[L].hd;
    t->cat    = malloc((size_t)2 * t->n_bb * sizeof(float));
    t->x      = malloc((size_t)t->n_inner * sizeof(float));
    t->h      = malloc((size_t)t->n_inner * sizeof(float));
    t->q      = malloc((size_t)q_max * sizeof(float));
    t->xb     = malloc((size_t)q_max * sizeof(float));
    t->o      = malloc((size_t)t->n_inner * sizeof(float));
    t->g1     = malloc((size_t)t->n_ff * sizeof(float));
    t->g2     = malloc((size_t)t->n_ff * sizeof(float));
    t->logits = malloc((size_t)t->n_vocab * sizeof(float));
    t->att    = NULL;
    t->att_cap = 0;
    if (!t->cat || !t->x || !t->h || !t->q || !t->xb || !t->o || !t->g1 || !t->g2 || !t->logits) {
        mtp_free(t);
        return NULL;
    }
    fprintf(stderr, "mtp: %s — %d blocks, %d wide, drafting for a %d-wide target\n",
            path, t->n_layer, t->n_inner, t->n_bb);
    return t;
}

void mtp_free(struct mtp *t) {
    if (!t) return;
    mtp_free_device(t);
    free_gguf(t->ctx);
    free(t->l);
    free(t->cat); free(t->x); free(t->h); free(t->q); free(t->xb);
    free(t->o); free(t->g1); free(t->g2); free(t->att); free(t->logits);
    free(t);
}

// ---- one draft step ----------------------------------------------------------

// Predict the token AFTER `token`, where `token` is the freshly chosen token
// that will sit at position `pos` (its own target forward need not have run
// yet). h_prev — the target's post-output-norm hidden from the forward that
// chose it — comes from the backend: model.last_hidden here, a device buffer
// on CUDA. The target cache holds positions < pos only — exactly the
// cross-attention this head was trained on.
int mtp_draft(struct mtp *t, const struct model *m, const struct kvcache *kv,
              int token, int pos) {
    if (!model_kv_host)
        return mtp_draft_device(t, m, kv, token, pos);
    const float *h_prev = m->last_hidden;
    const struct config *c = &m->cfg;
    const int ni = t->n_inner, nb = t->n_bb;
    const float eps = t->eps;

    if (pos + 1 > t->att_cap) {
        free(t->att);
        t->att_cap = pos + 256;
        t->att = malloc((size_t)t->att_cap * sizeof(float));
        if (!t->att) return -1;
    }

    // target's embedding of `token`, sqrt-scaled exactly like the main input
    float *erow = dequantize_row(gguf_find_tensor(m->ctx, "token_embd.weight"), token, nb);
    if (!erow) return -1;
    float sc = sqrtf((float)nb);
    for (int i = 0; i < nb; i++) t->cat[i] = erow[i] * sc;
    free(erow);
    memcpy(t->cat + nb, h_prev, (size_t)nb * sizeof(float));

    matmul_q(t->x, t->pre, t->cat, 2 * nb, ni);

    for (int L = 0; L < t->n_layer; L++) {
        const struct mtp_layer *bl = &t->l[L];
        const int hd = bl->hd, src = bl->src;
        const int q_dim = t->n_head * hd;
        const int kv_dim = kv->kv_dim[src], n_head_kv = m->n_head_kv[src];
        const int gqa = t->n_head / n_head_kv;
        const int seq = kv->seq[src], kf16 = kv->f16[src];
        const void *Kc = kv->k[src], *Vc = kv->v[src];

        // ---- cross-attention into the target's cache ----
        rmsnorm(t->h, t->x, bl->attn_norm, ni, eps);
        matmul_q(t->q, bl->q, t->h, ni, q_dim);
        for (int hh = 0; hh < t->n_head; hh++) {
            rmsnorm(t->q + hh * hd, t->q + hh * hd, bl->q_norm, hd, eps);
            rope_neox(t->q + hh * hd, hd, pos, bl->local ? t->base_swa : t->base_full);
        }

        // the cache holds positions [0, pos); SWA admits only the window back
        // from this step's position, same as the target's own attention
        int start = (bl->local && c->sliding_window > 0 && pos - c->sliding_window + 1 > 0)
                  ? pos - c->sliding_window + 1 : 0;
        for (int hh = 0; hh < t->n_head; hh++) {
            const float *qh = t->q + hh * hd;
            int kvh = hh / gqa;
            for (int p = start; p < pos; p++) {
                size_t off = (size_t)(p % seq) * kv_dim + (size_t)kvh * hd;
                float s = 0.0f;
                for (int i = 0; i < hd; i++) s += qh[i] * kv_at(Kc, off + i, kf16);
                t->att[p] = s;                          // scale 1.0, like the target
            }
            softmax(t->att + start, pos - start);
            float *outh = t->xb + hh * hd;
            for (int i = 0; i < hd; i++) outh[i] = 0.0f;
            for (int p = start; p < pos; p++) {
                size_t off = (size_t)(p % seq) * kv_dim + (size_t)kvh * hd;
                float a = t->att[p];
                for (int i = 0; i < hd; i++) outh[i] += a * kv_at(Vc, off + i, kf16);
            }
        }
        matmul_q(t->o, bl->o, t->xb, q_dim, ni);
        rmsnorm(t->o, t->o, bl->post_attn, ni, eps);
        for (int i = 0; i < ni; i++) t->h[i] = t->x[i] + t->o[i];   // attn_out

        // ---- gated FFN ----
        rmsnorm(t->x, t->h, bl->ffn_norm, ni, eps);
        matmul_q(t->g1, bl->gate, t->x, ni, t->n_ff);
        matmul_q(t->g2, bl->up,   t->x, ni, t->n_ff);
        for (int i = 0; i < t->n_ff; i++) t->g1[i] = gelu(t->g1[i]) * t->g2[i];
        matmul_q(t->o, bl->down, t->g1, t->n_ff, ni);
        rmsnorm(t->o, t->o, bl->post_ffw, ni, eps);
        for (int i = 0; i < ni; i++) t->x[i] = t->h[i] + t->o[i];
        if (bl->out_scale)
            for (int i = 0; i < ni; i++) t->x[i] *= bl->out_scale[0];
    }

    rmsnorm(t->x, t->x, t->out_norm, ni, eps);
    // (chaining via the post-projection comes with deeper spec blocks; at
    // block 2 there is exactly one draft per round, so it never runs)
    matmul_q(t->logits, t->head, t->x, ni, t->n_vocab);
    if (t->softcap > 0.0f)
        for (int v = 0; v < t->n_vocab; v++) t->logits[v] = t->softcap * tanhf(t->logits[v] / t->softcap);

    int best = 0;
    for (int v = 1; v < t->n_vocab; v++)
        if (t->logits[v] > t->logits[best]) best = v;
    return best;
}
