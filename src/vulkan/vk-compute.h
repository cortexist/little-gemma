// The Vulkan compute seam under model-vulkan.c: instance/device bring-up, the
// per-quant-type matmul pipelines (SPIR-V compiled from shaders/matmul.comp and
// embedded at build time), and weight residency. On an integrated GPU the GGUF
// blob is imported IN PLACE via VK_EXT_external_memory_host — zero-copy, the
// same trick the Orin CUDA build uses — so the weights exist once in memory;
// where import is unavailable each tensor is uploaded on first use instead
// (double residency: fine on a discrete card, tight on a 16 GB APU laptop).
// Not public API — that is include/model.h.
#ifndef VK_COMPUTE_H
#define VK_COMPUTE_H

#include "gguf.h"

// Bring up Vulkan and register the GGUF blob for zero-copy import. Returns 0
// on success, -1 if no usable device (the caller then stays on the host
// matmul). Idempotent; prints one line saying which device and weight path.
int  vkc_init(const struct gguf_context *ctx);
void vkc_destroy(void);
int  vkc_ready(void);

// out[m] = W . x with W quantized, on the GPU. Blocking (submit + fence): the
// caller's next op needs the result anyway — batching independent dispatches
// is the known next step, measured before built (see the performance journal
// culture). Returns 0, or -1 when this weight type has no shader / Vulkan is
// not up, and the caller falls back to the host path.
int  vkc_matmul(float *out, const struct gguf_tensor *t, const float *x, int k, int m);

#endif // VK_COMPUTE_H
