// No GPU backend in this build. media.c calls v_embed_image_gpu / v_gpu_free
// unconditionally (no #ifdef), so a target without a GPU encoder (the CPU `run`)
// links these no-op stubs: the GPU path returns NULL and media.c falls back to its
// host encoder. A GPU target links its implementation (src/cuda/media-cuda.cu)
// instead, which provides the real ones. (Same select-by-linked-file seam as
// model-cpu.c vs src/cuda/model-cuda-*.cu.)
#include "media-internal.h"

float *v_embed_image_gpu(struct media *md, const uint8_t *rgb, int w, int h, int *n_tokens) {
    (void)md; (void)rgb; (void)w; (void)h; (void)n_tokens;
    return NULL;   // no GPU here -> caller uses the host path
}

float *uv_embed_image_gpu(struct media *md, const uint8_t *rgb, int w, int h, int *n_tokens) {
    (void)md; (void)rgb; (void)w; (void)h; (void)n_tokens;
    return NULL;
}

float *uv_embed_audio_gpu(struct media *md, const int16_t *pcm, int n_samples, int *n_tokens) {
    (void)md; (void)pcm; (void)n_samples; (void)n_tokens;
    return NULL;
}

void v_gpu_free(struct media *md) { (void)md; }
