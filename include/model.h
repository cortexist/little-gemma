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
    float                     *last_hidden; // post-output-norm hidden of the most
                                            // recent logits-producing forward — what
                                            // the MTP draft head feeds on (n_embd;
                                            // the CPU backend fills it each forward)
};

int  model_init(struct model *m, const struct gguf_context *ctx);
void model_free(struct model *m);

// Rolling key/value cache. Head size varies per layer (local vs global), and
// only KV-owning layers allocate storage: a layer that reuses another's KV
// holds NULL. Sliding-window (local) layers only ever attend to the last
// `sliding_window` positions, so their storage is a RING of about that many
// rows — row for position p is p % seq[L] — instead of max_seq rows. Global
// layers keep the full max_seq (seq[L] == max_seq, and the ring index is then
// the identity). At a long context the cache cost is dominated by the few
// global KV-owning layers, not the layer count.
// Global layers store their rows as f16: at a long context the cache cost —
// capacity and the per-token read — is almost entirely theirs, and halving it
// is worth one round-to-nearest per stored value (the only step in this
// project's pipeline that changes the KV numbers; everything else is exact or
// reassociated). The sliding-window rings stay f32: a few hundred rows save
// nothing meaningful in f16, so they keep the exact values.
struct kvcache {
    int     n_layer;
    int     max_seq;   // logical capacity (positions); ring rows may be fewer
    int    *kv_dim;    // per layer: n_head_kv * head_dim(layer)
    int    *seq;       // per layer: rows allocated (ring length); 0 if reusing
    int    *f16;       // per layer: 1 if rows are stored as f16 (global layers)
    void  **k;         // per layer: [seq * kv_dim] f32 or f16, NULL if reusing
    void  **v;
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

// The forward for prompt tokens whose outputs nobody reads: fills the kv cache
// for tokens[0..n) at positions pos0..pos0+n-1 and skips the head — the final
// norm, the n_vocab×n_embd output projection (~10% of a token's weight
// traffic), and the argmax/sync. Use it for every prompt token but the last,
// whose logits pick the first generated token. The CUDA backends process the
// span in fixed-size chunks so each weight matrix is read from memory once per
// chunk instead of once per token — prefill is bandwidth-bound, so that factor
// is most of its cost.
void model_prefill(struct model *m, struct kvcache *kv, const int *tokens, int n, int pos0);

// Prefill a span of PRE-COMPUTED embedding rows (media tokens from media.h):
// rows is [n][n_embd], row i enters the model at position pos0+i exactly as
// given — media embeddings are NOT sqrt(n_embd)-scaled (only real token
// lookups are; the reference scales by `ubatch.token ? sqrt(n_embd) : 1`).
// On models with per-layer inputs (PLE) a media position takes the padding
// token's (id 0) per-layer row beside the usual projection of its embedding,
// matching the reference; the 12B has no PLE at all.
void model_prefill_embd(struct model *m, struct kvcache *kv, const float *rows, int n, int pos0);

// ---- MTP: the gemma4-assistant draft head (src/mtp.c) ----------------------
// A tiny transformer that predicts the token AFTER next by cross-attending
// straight into the target's KV cache (it has no K/V projections of its own).
// Greedy verify means drafts NEVER change the output — only how many target
// forwards run per emitted token. Host implementation: needs the cache rows
// in host memory (model_kv_host == 1; the CPU backend).
struct mtp;
struct mtp *mtp_open(const char *path, const struct model *m);
void        mtp_free(struct mtp *t);
// Draft the successor of `token` — the freshly chosen token for position
// `pos`, whose own forward need not have run. The backend supplies the other
// half of the head's input itself: the hidden state of the forward that chose
// `token` (CPU: model.last_hidden; CUDA: kept on the device).
int mtp_draft(struct mtp *t, const struct model *m, const struct kvcache *kv,
              int token, int pos);

// Verify a draft: feed tok0 at pos and tok1 (the draft) at pos+1 in one
// batched step. out[0] = greedy successor of tok0 (always valid); out[1] =
// successor of tok1 (valid only when out[0] == tok1, i.e. the draft held).
// Returns tokens advanced: 2 on accept, 1 on reject. The backend leaves its
// h_prev at the last valid position either way, so drafting chains. With
// greedy verification the emitted text is IDENTICAL to plain greedy decoding
// — only the number of forwards per token changes.
int model_forward2(struct model *m, struct kvcache *kv, int tok0, int tok1, int pos, int *out);

// 1 if this backend's kvcache rows live in host memory (the CPU backend);
// the CUDA backends keep them — and the draft head — on the device.
extern const int model_kv_host;

#endif // MODEL_H
