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
    void  **px_k;      // saved system-prefix rows (see kvcache_save_prefix), or NULL
    void  **px_v;
};

int  kvcache_init(struct kvcache *kv, const struct model *m, int max_seq);
void kvcache_free(struct kvcache *kv);

// The system-prefix trick that rescues TTFT for skills-in-context serving:
// prefill the skills turn ONCE at server start, save its cache rows, and
// restore them at each session start so every conversation begins at position
// n already knowing the skills — instead of re-prefilling them per session.
// Global layers' rows below n can never be overwritten (sessions only write
// at their own positions), but the sliding-window RINGS wrap during a long
// session and clobber prefix rows — restore repairs them. Per owning layer
// the saved state is the first min(n, seq) rows, which is the exact ring
// content at position n whether or not the prefix itself wrapped.
void kvcache_save_prefix(struct kvcache *kv, int n);
void kvcache_restore_prefix(struct kvcache *kv, int n);

// Run one token at sequence position `pos` (0-based). Reads/writes the kv cache
// and writes cfg.n_vocab logits into `logits` (caller-allocated). This is the
// full Gemma 4 decoder (per-layer-input embeddings, elastic FFN, KV sharing,
// softcap); logits match the llama.cpp reference.
void model_forward(struct model *m, struct kvcache *kv, int token, int pos, float *logits);

// Same forward, but returns argmax(logits) — the greedy next token — instead of
// the logits themselves. On the GPU backends the argmax runs on the device, so
// 4 bytes cross the bus per token instead of the whole vocabulary's logits.
int model_forward_next(struct model *m, struct kvcache *kv, int token, int pos);

// Optional pick hook (-temp): when set, every backend picks generated tokens by
// calling it on the softcapped logits row instead of taking the argmax — plain
// decode and the MTP verify alike (the draft head still drafts greedily; drafts
// only gate batching, the pick is what lands). NULL, the default, is greedy and
// byte-identical. On the GPU backends a set hook costs one n_vocab logits copy
// per generated token.
extern int (*model_pick)(const float *logits, int n);

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

// Prefill a whole MIXED span — text tokens and media rows interleaved — as
// one stream, so chunk boundaries ignore the text/media seams. ids[i] >= 0 is
// a text token (embedding looked up and sqrt(n_embd)-scaled, per-layer row
// from the id); ids[i] < 0 is media row -ids[i]-1 of rows (entered as given,
// per-layer row id 0). This exists because a turn prefilled as separate
// text/media/text calls pays a full weight pass for each segment's padded
// remainder — a short camera turn spent a third of its prefill on the seams.
void model_prefill_mixed(struct model *m, struct kvcache *kv, const float *rows,
                         const int *ids, int n, int pos0);

// Tell the engine a media projector is loaded, so it sizes the prefill activation
// buffers for a whole image span (up to the model's patch budget) before the decode
// graph captures their pointers. Call once at startup after media_open, before the
// first forward. LG_PREFILL_MAX_B caps the width (a deployment's VRAM throttle).
void model_prefill_reserve(void);

// MTP speculation block depth (verify width), a COMPILE-TIME constant. DEFAULT 3: a
// chained 2nd draft + triple verify, 1.1-1.3x over block-2 on 12B (A5000 and Orin),
// output byte-identical. Build -DLG_MTP_N=2 for the conservative one-draft pair verify;
// N generalizes. Recompile to change.
#ifndef LG_MTP_N
#define LG_MTP_N 3
#endif

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

// Chained draft for LG_MTP_N>2: draft the token AFTER a draft, on the draft head's
// own hidden (the target never ran this position, so its hidden isn't available — a
// chained draft is inherently weaker). Returns -1 where unsupported (CPU backend, or
// no post-projection); the caller then pads that slot so the verify just rejects it.
int mtp_draft_chain(struct mtp *t, const struct model *m, const struct kvcache *kv,
                    int token, int pos);

// Verify a block of LG_MTP_N tokens in one batched step: toks[0] is the freshly
// chosen token (at pos), toks[1..] the drafts (at pos+1..pos+LG_MTP_N-1). out[j] =
// greedy successor of toks[j]; out[0] is always valid, out[j] valid only when every
// earlier draft held. Returns tokens advanced (1..LG_MTP_N: the run of drafts that
// held, +1). The backend leaves h_prev at the last valid position so drafting chains.
// Greedy verification keeps the emitted text IDENTICAL to plain greedy decoding —
// only the number of target forwards per token changes.
int model_forward_spec(struct model *m, struct kvcache *kv, const int *toks, int pos, int *out);

// 1 if this backend's kvcache rows live in host memory (the CPU backend);
// the CUDA backends keep them — and the draft head — on the device.
extern const int model_kv_host;

#endif // MODEL_H
