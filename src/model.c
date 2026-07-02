// Common (backend-agnostic) model setup: reads hyperparameters and per-layer
// geometry from the GGUF metadata. The compute (kernels + forward) lives in a
// backend file picked at build time: model-cpu.c or model-cuda-{f32,i8}.cu.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"

// Build "<arch>.<suffix>" and read it as a u32 (with fallback).
static uint32_t arch_u32(const struct gguf_context *ctx, const char *arch,
                         const char *suffix, uint32_t fb) {
    char key[128];
    snprintf(key, sizeof key, "%s.%s", arch, suffix);
    return gguf_get_u32(ctx, key, fb);
}

static float arch_f32(const struct gguf_context *ctx, const char *arch,
                      const char *suffix, float fb) {
    char key[128];
    snprintf(key, sizeof key, "%s.%s", arch, suffix);
    return gguf_get_f32(ctx, key, fb);
}

// feed_forward_length is a per-layer i32 array in gemma (elastic FFN); return the
// max for buffer sizing. Falls back to a scalar for other architectures.
static int arch_ff(const struct gguf_context *ctx, const char *arch) {
    char key[128];
    snprintf(key, sizeof key, "%s.feed_forward_length", arch);
    const struct gguf_meta *meta = gguf_find_meta(ctx, key);
    if (meta && meta->type == GGUF_TYPE_ARRAY &&
        meta->value.arr.type == GGUF_TYPE_INT32 && meta->value.arr.n > 0) {
        const int32_t *a = meta->value.arr.data;
        int mx = 0;
        for (uint64_t i = 0; i < meta->value.arr.n; i++) if (a[i] > mx) mx = a[i];
        return mx;
    }
    return (int)gguf_get_u32(ctx, key, 0);
}

// Read an int hparam that may be stored as a scalar or a per-layer array (take
// element 0). Some gemma4 hparams (head_count_kv, feed_forward_length) are arrays.
static int arch_int0(const struct gguf_context *ctx, const char *arch,
                     const char *suffix, int fb) {
    char key[128];
    snprintf(key, sizeof key, "%s.%s", arch, suffix);
    const struct gguf_meta *meta = gguf_find_meta(ctx, key);
    if (meta && meta->type == GGUF_TYPE_ARRAY &&
        meta->value.arr.type == GGUF_TYPE_INT32 && meta->value.arr.n > 0)
        return ((const int32_t *)meta->value.arr.data)[0];
    return (int)gguf_get_u32(ctx, key, (uint32_t)fb);
}

// Fill out[L] with each layer's feed-forward width (constant fallback otherwise).
static void load_ffn_lens(const struct gguf_context *ctx, const char *arch,
                          int *out, int n_layer, int fallback) {
    char key[128];
    snprintf(key, sizeof key, "%s.feed_forward_length", arch);
    const struct gguf_meta *meta = gguf_find_meta(ctx, key);
    const int32_t *a = (meta && meta->type == GGUF_TYPE_ARRAY &&
                        meta->value.arr.type == GGUF_TYPE_INT32) ? meta->value.arr.data : NULL;
    for (int L = 0; L < n_layer; L++)
        out[L] = (a && (uint64_t)L < meta->value.arr.n) ? a[L] : fallback;
}

int config_load(struct config *c, const struct gguf_context *ctx) {
    memset(c, 0, sizeof(*c));

    const char *arch = gguf_get_str(ctx, "general.architecture", "");
    snprintf(c->arch, sizeof(c->arch), "%s", arch);
    if (!arch[0]) {
        fprintf(stderr, "config: missing general.architecture\n");
        return -1;
    }

    c->n_layer           = (int)arch_u32(ctx, arch, "block_count", 0);
    c->n_embd            = (int)arch_u32(ctx, arch, "embedding_length", 0);
    c->n_head            = arch_int0(ctx, arch, "attention.head_count", 0);
    c->n_head_kv         = arch_int0(ctx, arch, "attention.head_count_kv", c->n_head);
    c->n_ff              = arch_ff(ctx, arch);
    c->n_ctx             = (int)arch_u32(ctx, arch, "context_length", 0);
    c->sliding_window    = (int)arch_u32(ctx, arch, "attention.sliding_window", 0);
    c->n_embd_per_layer  = (int)arch_u32(ctx, arch, "embedding_length_per_layer_input", 0);
    c->rms_eps           = arch_f32(ctx, arch, "attention.layer_norm_rms_epsilon", 1e-5f);
    c->rope_freq_base    = arch_f32(ctx, arch, "rope.freq_base", 10000.0f);
    c->rope_freq_base_swa= arch_f32(ctx, arch, "rope.freq_base_swa", c->rope_freq_base);
    c->logit_softcap     = arch_f32(ctx, arch, "final_logit_softcapping", 0.0f);

    // Per-layer head sizes: global (full) layers use key_length, sliding-window
    // layers use key_length_swa. Gemma4 differs between the two (512 vs 256).
    c->head_dim_full = (int)arch_u32(ctx, arch, "attention.key_length", 0);
    c->head_dim_swa  = (int)arch_u32(ctx, arch, "attention.key_length_swa", c->head_dim_full);

    // KV sharing: the first n_kv_start layers own their KV; later layers reuse it.
    int shared = (int)arch_u32(ctx, arch, "attention.shared_kv_layers", 0);
    c->n_kv_start = shared > 0 ? c->n_layer - shared : c->n_layer;

    // Vocabulary size = number of tokens in the tokenizer list.
    const struct gguf_meta *toks = gguf_find_meta(ctx, "tokenizer.ggml.tokens");
    if (toks && toks->type == GGUF_TYPE_ARRAY) c->n_vocab = (int)toks->value.arr.n;

    // Bounded, not just nonzero: block_count comes off disk as a u32, and a
    // garbage value cast to int could go NEGATIVE — which a later
    // (size_t)n_layer would sign-extend into an absurd allocation request
    // (GCC's -Walloc-size-larger-than spotted exactly that path). No real
    // model is within two orders of magnitude of the bound.
    if (c->n_layer <= 0 || c->n_layer > 4096) {
        fprintf(stderr, "config: %s.block_count is %d - missing or not a plausible layer count\n",
                arch, c->n_layer);
        return -1;
    }
    return 0;
}

void config_print(const struct config *c) {
    printf("--- config (%s) ---\n", c->arch);
    printf("n_layer        = %d\n", c->n_layer);
    printf("n_embd         = %d\n", c->n_embd);
    printf("n_head         = %d\n", c->n_head);
    printf("n_head_kv      = %d\n", c->n_head_kv);
    printf("head_dim_swa   = %d\n", c->head_dim_swa);
    printf("head_dim_full  = %d\n", c->head_dim_full);
    printf("n_ff           = %d\n", c->n_ff);
    printf("n_vocab        = %d\n", c->n_vocab);
    printf("n_ctx          = %d\n", c->n_ctx);
    printf("sliding_window = %d\n", c->sliding_window);
    printf("n_embd_per_lyr = %d\n", c->n_embd_per_layer);
    printf("n_kv_start     = %d\n", c->n_kv_start);
    printf("rms_eps        = %g\n", c->rms_eps);
    printf("rope_freq_base = %g (swa %g)\n", c->rope_freq_base, c->rope_freq_base_swa);
    printf("logit_softcap  = %g\n", c->logit_softcap);
}

// ---- model lifecycle (host metadata) --------------------------------------

int model_init(struct model *m, const struct gguf_context *ctx) {
    memset(m, 0, sizeof(*m));
    if (config_load(&m->cfg, ctx) != 0) return -1;
    m->ctx = ctx;
    // a LOCAL with a visible bound: config_load already validated the range,
    // but GCC's allocation-size analysis cannot see that through the struct
    // member, and warned that (size_t)<negative int> would be an absurd calloc
    const size_t n_layer = (size_t)(unsigned)m->cfg.n_layer;

    // Per-layer feed-forward widths (elastic FFN).
    m->ffn_len = calloc(n_layer, sizeof(int));
    if (!m->ffn_len) return -1;
    load_ffn_lens(ctx, m->cfg.arch, m->ffn_len, m->cfg.n_layer, m->cfg.n_ff);

    // Which layers use sliding-window (local) attention.
    m->is_local = calloc(n_layer, sizeof(int));
    if (!m->is_local) return -1;
    char key[128];
    snprintf(key, sizeof key, "%s.attention.sliding_window_pattern", m->cfg.arch);
    const struct gguf_meta *pattern = gguf_find_meta(ctx, key);
    if (pattern && pattern->type == GGUF_TYPE_ARRAY && pattern->value.arr.type == GGUF_TYPE_BOOL) {
        const int8_t *b = pattern->value.arr.data;
        for (int L = 0; L < m->cfg.n_layer && (uint64_t)L < pattern->value.arr.n; L++)
            m->is_local[L] = b[L] != 0;
    }

    // Per-layer head geometry, derived from the actual q/k tensor shapes — robust
    // across models where head_dim and head_count_kv vary by layer or are stored
    // as arrays (e.g. Gemma 12B has head_count_kv = [8, 8, ...]).
    m->head_dim  = calloc(n_layer, sizeof(int));
    m->n_head_kv = calloc(n_layer, sizeof(int));
    if (!m->head_dim || !m->n_head_kv) return -1;
    for (int L = 0; L < m->cfg.n_layer; L++) {
        char name[96];
        snprintf(name, sizeof name, "blk.%d.attn_q.weight", L);
        const struct gguf_tensor *q = gguf_find_tensor(ctx, name);
        snprintf(name, sizeof name, "blk.%d.attn_k.weight", L);
        const struct gguf_tensor *k = gguf_find_tensor(ctx, name);
        if (!q || m->cfg.n_head == 0 || (!k && L < m->cfg.n_kv_start)) {
            fprintf(stderr, "model: missing attn_q/attn_k for layer %d\n", L);
            return -1;
        }
        m->head_dim[L] = (int)(q->dims[1] / (uint64_t)m->cfg.n_head);
        if (k) {
            m->n_head_kv[L] = (int)(k->dims[1] / (uint64_t)m->head_dim[L]);
        } else {
            // A kv-shared layer with no attn_k: QAT exports prune the dead
            // k/v projections there (the layer only reads its source layer's
            // cache, and the forwards never compute k/v past n_kv_start).
            // It inherits the source layer's geometry — sources all precede
            // n_kv_start, so theirs is already derived.
            m->n_head_kv[L] = m->n_head_kv[m->cfg.n_kv_start - (m->is_local[L] ? 2 : 1)];
        }
    }

    // scratch for the MTP draft head's h_prev (filled by the backend's forward)
    m->last_hidden = calloc((size_t)m->cfg.n_embd, sizeof(float));
    if (!m->last_hidden) return -1;
    return 0;
}

void model_free(struct model *m) {
    if (!m) return;
    free(m->is_local);
    free(m->ffn_len);
    free(m->head_dim);
    free(m->n_head_kv);
    free(m->last_hidden);
    m->last_hidden = NULL;
    m->is_local = NULL;
    m->ffn_len = NULL;
    m->head_dim = NULL;
    m->n_head_kv = NULL;
}
