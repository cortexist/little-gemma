// Shared between media.c (host embedders) and media-kernel.cu (the GPU gemma4v
// encoder). Not part of the public API — that is include/media.h. The CPU-only
// build links a no-op stub (media-gpu-stub.c) for the GPU seam below; the CUDA
// targets link media-kernel.cu instead, and the host path stays in as oracle + fallback
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

// QAT fake-quant ranges (the reference's Gemma4ClippableLinear): activations
// clamp to [ilo,ihi] before and [olo,ohi] after a projection. The QAT-era
// mmprojs carry one quad per projection weight as f32 scalar tensors
// (<weight>.input_min/.input_max/.output_min/.output_max); running those
// weights WITHOUT the clamps is what turns speech into gibberish. Absent
// scalars default to +/-FLT_MAX — a no-op, which is why pre-QAT files behave
// exactly as before.
struct aclamp { float ilo, ihi, olo, ohi; };

// One block of the legacy (gemma4a) audio conformer.
struct alayer {
    const float *ffn_norm, *ffn_post, *ffn_norm1, *ffn_post1;
    const struct gguf_tensor *ffn_up, *ffn_down, *ffn_up1, *ffn_down1;
    const float *attn_pre, *attn_post, *pds;        // pds: per-dim Q scale [d_head]
    const struct gguf_tensor *q, *k, *v, *o, *k_rel;
    const float *norm_conv, *conv_norm, *ln2, *dw;  // dw: depthwise taps [a_embd][5]
    const struct gguf_tensor *pw1, *pw2;
    struct aclamp c_ffn_up, c_ffn_down, c_ffn_up1, c_ffn_down1;   // QAT ranges
    struct aclamp c_q, c_k, c_v, c_o, c_pw1, c_pw2;               // (k_rel has none)
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
    // audio, legacy (gemma4a, E2B/E4B): log-mel frontend + 12-block conformer.
    // Opened only under LG_GEMMA4A=1 — the QAT-era mmprojs transcribe, the
    // original encoder export confabulated (2026-07-19 isolation, memory of it
    // in the repo journals); default stays the Whisper lane.
    int legacy_a;
    int a_embd, a_head, a_layer, n_mel;        // 1024?, 8, 12, 128 (d_head = a_embd/a_head)
    const struct gguf_tensor *a_conv0, *a_conv1, *a_inp_proj, *a_out_proj;
    const float *a_conv0_n, *a_conv1_n, *a_out_proj_b;
    struct alayer *al;
    float *mel_filt;                           // [n_mel][n_fft/2+1] triangles
    float *rpe;                                // [13][a_embd] sinusoids
    void *audio_gpu;                           // GPU conformer state, or NULL
    void *gpu;                                 // GPU backend state (media-kernel.cu), or NULL
};

#ifdef __cplusplus
extern "C" {
#endif
// GPU embedders. Each returns NULL when the GPU path is unusable (init failure,
// an unsupported geometry) — the caller then falls back to the host path.
float *v_embed_image_gpu(struct media *md, const uint8_t *rgb, int w, int h, int *n_tokens);
float *uv_embed_image_gpu(struct media *md, const uint8_t *rgb, int w, int h, int *n_tokens);
float *uv_embed_audio_gpu(struct media *md, const int16_t *pcm, int n_samples, int *n_tokens);
// GPU gemma4a conformer: takes the host-built feature rows F[T][n_feat] (the
// log-mel + subsampling convs stay on the host) and runs the 12 blocks + the
// projections. NULL when CUDA is unusable — caller falls back to a_blocks_host.
float *a_blocks_gpu(struct media *md, const float *F, int T, int n_feat, int *n_tokens);
void   v_gpu_free(struct media *md);           // frees vision AND audio GPU state
#ifdef __cplusplus
}
#endif

#endif
