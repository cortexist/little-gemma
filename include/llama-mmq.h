// Vendored q4_K MMQ matmul from llama.cpp (see src/llama-mmq.cu). Unlike the
// flash-llama.cu bridge — which #includes mmq.cuh from a pinned llama checkout via
// -I — this carries llama's REAL kernel code (pruned to the q4_K path) self-contained
// in one .cu, with a hand-written ggml shim replacing common.cuh/ggml-common.h/ggml.h.
// No ggml -I needed. Bit-identical to the -I-based lg_mmq_q4k by construction.
#pragma once
#include <cuda_runtime.h>

// Route a q4_K prefill matmul O[n x m] = A[n x k] . W[m x k] through llama's mul_mat_q.
// w = block_q4_K in place; d_x = token-major f32 activation [n x k]; d_out [n x m].
void lg_mmq_q4k(float* d_out, const void* w, int k, int m, const float* d_x, int n, cudaStream_t stream);
