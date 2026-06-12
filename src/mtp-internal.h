// Shared between mtp.c (open/validate + the host draft) and the backends.
// Not public API — that is include/model.h. A backend whose KV cache lives on
// the device (CUDA) implements mtp_draft_device/mtp_free_device and keeps its
// own uploaded copy of these tensors behind `cuda`; the CPU backend stubs them.
#ifndef MTP_INTERNAL_H
#define MTP_INTERNAL_H

#include "gguf.h"
#include "model.h"

struct mtp_layer {
    const float *attn_norm, *q_norm, *post_attn, *ffn_norm, *post_ffw;
    const float *out_scale;                  // scalar
    const struct gguf_tensor *q, *o, *gate, *up, *down;
    int hd;            // q head dim, matches the target KV layer it reads
    int local;         // 1 = reads the target's SWA KV, 0 = global KV
    int src;           // target kv layer whose cache this block attends
};

struct mtp {
    struct gguf_context *ctx;
    int n_layer, n_inner, n_bb, n_vocab, n_head, n_ff;
    float eps, base_full, base_swa, softcap;
    const struct gguf_tensor *pre, *post, *head;   // next-token pre/post projections, LM head
    const float *out_norm;
    struct mtp_layer *l;
    // host-draft scratch (CPU backend; sized once at open)
    float *cat, *x, *h, *q, *xb, *o, *g1, *g2, *att, *logits;
    int att_cap;
    void *cuda;        // device state (model-cuda.cuh), or NULL
};

int  mtp_draft_device(struct mtp *t, const struct model *m, const struct kvcache *kv,
                      int token, int pos);
void mtp_free_device(struct mtp *t);

#endif // MTP_INTERNAL_H
