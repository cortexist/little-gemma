// The Vulkan compute seam under model-vulkan.c: instance/device bring-up, the
// per-quant-type matmul pipelines (SPIR-V compiled from shaders/matmul.comp and
// embedded at build time — one narrow NB=1 and one wide NB=16 per type), and
// weight residency. On an integrated GPU the GGUF blob is imported IN PLACE
// via VK_EXT_external_memory_host — zero-copy, the same trick the Orin CUDA
// build uses — so the weights exist once in memory; where import is
// unavailable each tensor is uploaded on first use instead (double residency:
// fine on a discrete card, tight on a 16 GB APU laptop).
// Not public API — that is include/model.h.
#ifndef VK_COMPUTE_H
#define VK_COMPUTE_H

#include "gguf.h"

#define VKC_NB_MAX   16   // columns the widest pipeline handles (shader NB)
#define VKC_NB_MID   4    // the small-batch pipeline (MTP verify, short tails)
#define VKC_TM       4    // rows per workgroup in the wide pipelines (shader TM)
#define VKC_MAX_JOBS 4    // dispatches per submit (q/k/v = 3, gate/up = 2)

// Bring up Vulkan and register the GGUF blob for zero-copy import. Returns 0
// on success, -1 if no usable device (the caller then stays on the host
// matmul). Idempotent; prints one line saying which device and weight path.
int  vkc_init(const struct gguf_context *ctx);
void vkc_destroy(void);
int  vkc_ready(void);
int  vkc_supported(uint32_t ggml_type);   // 1 if this weight type has pipelines

// One matmul of a batch: out[b*m + row] = W . x_column_b for b < nb.
struct vkc_job {
    const struct gguf_tensor *t;
    float                    *out;   // nb*m floats, column-major per column
    int                       m;
};

// Run njobs independent matmuls over ONE shared x (nb contiguous columns of
// k floats) as a single submit — one fence round-trip instead of njobs.
// q/k/v and gate/up are exactly this shape: same input, different weights.
// Blocking; the caller's next op needs the results anyway. Returns 0, or -1
// when any job's weight type has no shader / Vulkan is not up — the caller
// then takes the host path for the whole batch.
int vkc_matmul_x(const float *x, int k, int nb, const struct vkc_job *jobs, int njobs);

#endif // VK_COMPUTE_H
