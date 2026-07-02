#ifndef QUANT_H
#define QUANT_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "gguf.h"

// ggml tensor types — the values stored in gguf_tensor.type. These are ggml's
// own ids (and its block layouts), unavoidable like GGUF itself. Only the types
// this project handles are listed.
enum ggml_type {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_BF16 = 30,
};

// Short name ("f32", "q4_K", ...) or "?" if unknown.
const char *ggml_type_name(uint32_t type);

// Block geometry: elements per block, and bytes per block. Quantized types pack
// `blck_size` weights into `type_size` bytes; plain types use blck_size == 1.
// Both return 0 for an unsupported type.
int    ggml_blck_size(uint32_t type);
size_t ggml_type_size(uint32_t type);

// Bytes that `n_elements` of `type` occupy on disk. 0 if unsupported or if
// n_elements is not a whole number of blocks.
size_t ggml_nbytes(uint32_t type, int64_t n_elements);

// Dequantize a tensor's data to a freshly malloc'd f32 array of
// product(dims) values. Returns NULL on an unsupported type or OOM.
// The caller frees the result.
float *dequantize(const struct gguf_tensor *t);

// Dequantize one logical row (row_len elements starting at element row*row_len)
// to a fresh f32 array. Avoids materializing a huge table when only one row is
// needed. Start and row_len must be block-aligned. Caller frees.
float *dequantize_row(const struct gguf_tensor *t, int64_t row, int64_t row_len);

// Dequantize `n` elements of `type` from `src` into the caller's `out` buffer.
// n must be a whole number of blocks. Returns false for an unsupported type.
bool dequantize_into(uint32_t type, const void *src, float *out, int64_t n);

#endif // QUANT_H
