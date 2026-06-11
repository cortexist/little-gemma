#ifndef MODEL_H
#define MODEL_H

#include "gguf.h"

// The hyperparameters that shape the compute graph, pulled out of the GGUF
// metadata into plain typed fields. The metadata keys are prefixed by the
// architecture name (e.g. "gemma4.block_count"); config_load reads that prefix
// from general.architecture so this works for any llama-family arch.
struct config {
    char  arch[32];           // general.architecture (e.g. "gemma4")
    int   n_layer;            // number of transformer blocks
    int   n_embd;             // embedding / model dimension
    int   n_head;             // query heads
    int   n_head_kv;          // key/value heads (< n_head for grouped-query)
    int   head_dim_swa;       // head size on sliding-window (local) layers
    int   head_dim_full;      // head size on full (global) layers
    int   n_ff;               // feed-forward hidden size (per-layer; assumed uniform)
    int   n_vocab;            // vocabulary size
    int   n_ctx;              // max context length the file was trained for
    int   sliding_window;     // local-attention window
    int   n_embd_per_layer;   // per-layer-input embedding size (PLE)
    int   n_kv_start;         // layers [0, n_kv_start) own KV; later layers reuse
    float rms_eps;            // RMS normalization epsilon
    float rope_freq_base;     // RoPE base on full layers
    float rope_freq_base_swa; // RoPE base on sliding-window layers
    float logit_softcap;      // final logit soft-cap (0 if none)
};

// Fill `c` from `ctx`. Returns 0 on success, -1 if the file looks unusable
// (no architecture or zero layers).
int  config_load(struct config *c, const struct gguf_context *ctx);
void config_print(const struct config *c);

// A loaded model ready to run. Weights stay quantized in `ctx`; each row is
// dequantized on the fly inside forward() (nothing is cached in f32).
struct model {
    struct config              cfg;
    const struct gguf_context *ctx;       // borrowed; holds the quantized weights
    int                       *is_local;   // per layer: 1 = sliding-window attention
    int                       *ffn_len;    // per layer: feed-forward width (elastic FFN)
    int                       *head_dim;   // per layer: size of one attention head
    int                       *n_head_kv;  // per layer: number of key/value heads
};

int  model_init(struct model *m, const struct gguf_context *ctx);
void model_free(struct model *m);

// Rolling key/value cache. Head size varies per layer (local vs global), and
// only KV-owning layers allocate storage, so each layer has its own buffer.
struct kvcache {
    int     n_layer;
    int     max_seq;
    int    *kv_dim;    // per layer: n_head_kv * head_dim(layer)
    float **k;         // per layer: [max_seq * kv_dim], or NULL if the layer reuses KV
    float **v;
};

int  kvcache_init(struct kvcache *kv, const struct model *m, int max_seq);
void kvcache_free(struct kvcache *kv);

// Run one token at sequence position `pos` (0-based). Reads/writes the kv cache
// and writes cfg.n_vocab logits into `logits` (caller-allocated). This is the
// full Gemma 4 decoder (per-layer-input embeddings, elastic FFN, KV sharing,
// softcap); logits match the llama.cpp reference.
void model_forward(struct model *m, struct kvcache *kv, int token, int pos, float *logits);

// Same forward, but returns argmax(logits) — the greedy next token — instead of
// the logits themselves. On the GPU backends the argmax runs on the device, so
// 4 bytes cross the bus per token instead of the whole vocabulary's logits.
int model_forward_next(struct model *m, struct kvcache *kv, int token, int pos);

// The forward for a prompt token whose output nobody reads: fills the kv cache
// and skips the head — the final norm, the n_vocab×n_embd output projection
// (~10% of a token's weight traffic), and the argmax/sync. Use it for every
// prompt token but the last, whose logits pick the first generated token.
void model_prefill(struct model *m, struct kvcache *kv, int token, int pos);

#endif // MODEL_H
