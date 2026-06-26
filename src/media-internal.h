// Shared between media.c (host embedders) and media-cuda.cu (the GPU gemma4v
// encoder). Not part of the public API — that is include/media.h. The CPU-only
// build links a no-op stub (media-gpu-stub.c) for the GPU seam below; the CUDA
// targets link media-cuda.cu instead, and the host path stays in as oracle + fallback
// (set LG_MEDIA_VERIFY=1 to run both per image and print the max difference).
#ifndef MEDIA_INTERNAL_H
#define MEDIA_INTERNAL_H

#include <stdint.h>
#include "gguf.h"

// One block of the legacy (gemma4v) vision transformer.
struct vlayer {
    const float *ln1, *ln2, *attn_post, *ffn_post;   // rms norm weights [768]
    const float *qn, *kn;                            // per-head q/k norms [64]
    const struct gguf_tensor *q, *k, *v, *o;         // [768 -> 768], no biases
    const struct gguf_tensor *gate, *up, *down;      // gated FFN, 768<->3072
};

struct media {
    struct gguf_context *ctx;
    int n_embd;                    // LLM width; every output row is this long
    // vision, unified (gemma4uv): one linear over patch x patch pixel blocks
    int patch;                     // pixels per patch side (48: 16 x merge 3)
    int pos_size;                  // rows in each axis of the position table
    const struct gguf_tensor *v_patch_w;       // [patch*patch*3 -> n_embd]
    const float *v_patch_b;
    const float *vn1w, *vn1b;                  // LayerNorm before the embed
    const float *vn2w, *vn2b;                  // LayerNorm after it
    const float *vn3w, *vn3b;                  // LayerNorm after the pos add
    const float *v_pos;                        // [n_embd, pos_size, 2]: x table, then y
    const struct gguf_tensor *mm_v;            // [enc_width -> n_embd]
    // audio, unified (gemma4ua): one linear over raw 640-sample frames
    int frame;                                 // samples per frame (640 = 40 ms @ 16 kHz)
    const struct gguf_tensor *mm_a;            // [frame -> n_embd]
    // vision, legacy (gemma4v, E2B/E4B): a 16-block ViT + 3x3 average pool
    int legacy_v;
    int v_embd, v_head, v_layer, v_merge;      // 768, 12, 16, 3
    const struct gguf_tensor *v_patch16;       // [16,16,3 -> 768] conv, no bias
    struct vlayer *vl;
    void *gpu;                                 // GPU backend state (media-cuda.cu), or NULL
};

#ifdef __cplusplus
extern "C" {
#endif
// GPU gemma4v encoder. Returns NULL when the GPU path is unusable (init failure, an
// unsupported geometry) — the caller then falls back to the host path.
float *v_embed_image_gpu(struct media *md, const uint8_t *rgb, int w, int h, int *n_tokens);
void   v_gpu_free(struct media *md);
#ifdef __cplusplus
}
#endif

#endif
