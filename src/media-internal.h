// Shared between media.c (host embedders) and media-cuda.cu (the GPU gemma4v
// encoder). Not part of the public API — that is include/media.h. The CPU-only
// build compiles media.c alone and never defines LG_MEDIA_CUDA; the CUDA
// targets add media-cuda.cu and the host path stays in as oracle + fallback
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

// One block of the legacy (gemma4a) audio conformer.
struct alayer {
    const float *ffn_norm, *ffn_post, *ffn_norm1, *ffn_post1;
    const struct gguf_tensor *ffn_up, *ffn_down, *ffn_up1, *ffn_down1;
    const float *attn_pre, *attn_post, *pds;        // pds: per-dim Q scale [d_head]
    const struct gguf_tensor *q, *k, *v, *o, *k_rel;
    const float *norm_conv, *conv_norm, *ln2, *dw;  // dw: depthwise taps [a_embd][5]
    const struct gguf_tensor *pw1, *pw2;
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
    // audio, legacy (gemma4a, E2B/E4B): log-mel frontend + 12-block conformer
    int legacy_a;
    int a_embd, a_head, a_layer, n_mel;        // 1024, 8, 12, 128 (d_head = a_embd/a_head)
    const struct gguf_tensor *a_conv0, *a_conv1, *a_inp_proj, *a_out_proj;
    const float *a_conv0_n, *a_conv1_n, *a_out_proj_b;
    struct alayer *al;
    float *mel_filt;                           // [n_mel][n_fft/2+1] triangles
    float *rpe;                                // [13][a_embd] sinusoids
    void *audio_cuda;                          // GPU conformer state (Phase 2), or NULL
    void *cuda;                                // media-cuda.cu (vision) state, or NULL
};

#ifdef __cplusplus
extern "C" {
#endif
// GPU gemma4v encoder. Returns NULL when CUDA is unusable (init failure, an
// unsupported geometry) — the caller then falls back to the host path.
float *v_embed_image_cuda(struct media *md, const uint8_t *rgb, int w, int h, int *n_tokens);
void   v_cuda_free(struct media *md);

// GPU gemma4a conformer. Takes the host-built feature rows F[T][Fdim] (the
// log-mel + subsampling convs stay on the host) and runs the 12 blocks + the
// projections on the GPU. Returns NULL when CUDA is unusable — caller falls
// back to a_blocks_host on the same F.
float *a_blocks_cuda(struct media *md, const float *F, int T, int n_feat, int *n_tokens);
void   a_cuda_free(struct media *md);
#ifdef __cplusplus
}
#endif

#endif
