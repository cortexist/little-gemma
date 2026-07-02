#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "quant.h"

// Super-block size shared by all K-quants.
#define QK_K 256

typedef uint16_t ggml_half;

// ---- half / bfloat to float ----------------------------------------------

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                       // +/- zero
        } else {                               // subnormal half -> normal float
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {                  // inf / nan
        bits = sign | 0x7F800000u | (mant << 13);
    } else {                                   // normal
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static float bf16_to_fp32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;         // bf16 is the top 16 bits of f32
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// ---- block layouts (must match ggml byte-for-byte) ------------------------

#define K_SCALE_SIZE 12

typedef struct {                 // Q3_K: 110 bytes
    uint8_t   hmask[QK_K/8];     // 3rd (high) bit of each 3-bit weight
    uint8_t   qs[QK_K/4];        // low 2 bits, packed 4 per byte
    uint8_t   scales[12];        // 16 x 6-bit scales, packed
    ggml_half d;                 // super-block scale
} block_q3_K;

typedef struct {                 // Q4_K: 144 bytes
    ggml_half d;                 // scale of the 6-bit block scales
    ggml_half dmin;              // scale of the 6-bit block mins
    uint8_t   scales[K_SCALE_SIZE];
    uint8_t   qs[QK_K/2];        // 4-bit weights
} block_q4_K;

typedef struct {                 // Q5_K: 176 bytes
    ggml_half d;
    ggml_half dmin;
    uint8_t   scales[K_SCALE_SIZE];
    uint8_t   qh[QK_K/8];        // 5th bit of each weight
    uint8_t   qs[QK_K/2];        // low 4 bits
} block_q5_K;

typedef struct {                 // Q6_K: 210 bytes
    uint8_t   ql[QK_K/2];        // low 4 bits
    uint8_t   qh[QK_K/4];        // high 2 bits
    int8_t    scales[QK_K/16];   // 16 x int8 block scales
    ggml_half d;                 // super-block scale
} block_q6_K;

typedef struct {                 // Q8_0: 34 bytes
    ggml_half d;
    int8_t    qs[32];
} block_q8_0;

typedef struct {                 // Q4_0: 18 bytes — the QAT release format
    ggml_half d;
    uint8_t   qs[16];            // element j in the low nibble, j+16 in the high
} block_q4_0;

_Static_assert(sizeof(block_q4_0) ==  18, "block_q4_0 layout");
_Static_assert(sizeof(block_q3_K) == 110, "block_q3_K layout");
_Static_assert(sizeof(block_q4_K) == 144, "block_q4_K layout");
_Static_assert(sizeof(block_q5_K) == 176, "block_q5_K layout");
_Static_assert(sizeof(block_q6_K) == 210, "block_q6_K layout");
_Static_assert(sizeof(block_q8_0) ==  34, "block_q8_0 layout");

// ---- traits ---------------------------------------------------------------

const char *ggml_type_name(uint32_t type) {
    switch (type) {
        case GGML_TYPE_F32:  return "f32";
        case GGML_TYPE_F16:  return "f16";
        case GGML_TYPE_Q4_0: return "q4_0";
        case GGML_TYPE_Q8_0: return "q8_0";
        case GGML_TYPE_Q2_K: return "q2_K";
        case GGML_TYPE_Q3_K: return "q3_K";
        case GGML_TYPE_Q4_K: return "q4_K";
        case GGML_TYPE_Q5_K: return "q5_K";
        case GGML_TYPE_Q6_K: return "q6_K";
        case GGML_TYPE_BF16: return "bf16";
        default:             return "?";
    }
}

int ggml_blck_size(uint32_t type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16: return 1;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0: return 32;
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K: return QK_K;
        default:             return 0;
    }
}

size_t ggml_type_size(uint32_t type) {
    switch (type) {
        case GGML_TYPE_F32:  return 4;
        case GGML_TYPE_F16:  return 2;
        case GGML_TYPE_BF16: return 2;
        case GGML_TYPE_Q4_0: return sizeof(block_q4_0);
        case GGML_TYPE_Q8_0: return sizeof(block_q8_0);
        case GGML_TYPE_Q3_K: return sizeof(block_q3_K);
        case GGML_TYPE_Q4_K: return sizeof(block_q4_K);
        case GGML_TYPE_Q5_K: return sizeof(block_q5_K);
        case GGML_TYPE_Q6_K: return sizeof(block_q6_K);
        default:             return 0;
    }
}

size_t ggml_nbytes(uint32_t type, int64_t n_elements) {
    int blck = ggml_blck_size(type);
    if (blck == 0 || n_elements < 0 || n_elements % blck != 0) return 0;
    return (size_t)(n_elements / blck) * ggml_type_size(type);
}

// ---- dequantization kernels ----------------------------------------------
//
// These mirror ggml's dequantize_row_* exactly; the bit-twiddling is what packs
// 256 weights into ~110-176 bytes plus a couple of fp16 scales.

// 6-bit (scale, min) pair for sub-block j, unpacked from the 12-byte `scales`.
static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>   4) | ((q[j - 0] >> 6) << 4);
    }
}

static void dequant_q4_K(const block_q4_K *x, float *y, int nb) {
    for (int i = 0; i < nb; i++) {
        const float d   = fp16_to_fp32(x[i].d);
        const float min = fp16_to_fp32(x[i].dmin);
        const uint8_t *q = x[i].qs;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4)  - m2;
            q += 32; is += 2;
        }
    }
}

static void dequant_q5_K(const block_q5_K *x, float *y, int nb) {
    for (int i = 0; i < nb; i++) {
        const float d   = fp16_to_fp32(x[i].d);
        const float min = fp16_to_fp32(x[i].dmin);
        const uint8_t *ql = x[i].qs;
        const uint8_t *qh = x[i].qh;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4)  + ((qh[l] & u2) ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

static void dequant_q3_K(const block_q3_K *x, float *y, int nb) {
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    const int8_t *scales = (const int8_t *)aux;

    for (int i = 0; i < nb; i++) {
        const float d_all = fp16_to_fp32(x[i].d);
        const uint8_t *q  = x[i].qs;
        const uint8_t *hm = x[i].hmask;
        uint8_t m = 1;

        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = ((aux[0] >> 0) & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = ((aux[1] >> 0) & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                shift += 2;
                m <<= 1;   // high-bit selector advances per sub-block (8 bits/byte)
            }
            q += 32;
        }
    }
}

static void dequant_q6_K(const block_q6_K *x, float *y, int nb) {
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t  *sc = x[i].scales;
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

static void dequant_q8_0(const block_q8_0 *x, float *y, int nb) {
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int l = 0; l < 32; ++l) *y++ = d * x[i].qs[l];
    }
}

static void dequant_q4_0(const block_q4_0 *x, float *y, int nb) {
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int l = 0; l < 16; ++l) {
            y[l]      = d * (int)((x[i].qs[l] & 0xF) - 8);
            y[l + 16] = d * (int)((x[i].qs[l] >>  4) - 8);
        }
        y += 32;
    }
}

// Dequantize `n` elements of `type` from `src` into `y`. `n` must be a whole
// number of blocks. Returns false for an unsupported type.
static bool deq_n(uint32_t type, const void *src, float *y, int64_t n) {
    int blck = ggml_blck_size(type);
    if (blck == 0 || n % blck != 0) return false;
    int nb = (int)(n / blck);
    switch (type) {
        case GGML_TYPE_F32: memcpy(y, src, (size_t)n * sizeof(float)); break;
        case GGML_TYPE_F16: {
            const uint16_t *h = src;
            for (int64_t i = 0; i < n; i++) y[i] = fp16_to_fp32(h[i]);
            break;
        }
        case GGML_TYPE_BF16: {
            const uint16_t *h = src;
            for (int64_t i = 0; i < n; i++) y[i] = bf16_to_fp32(h[i]);
            break;
        }
        case GGML_TYPE_Q4_0: dequant_q4_0(src, y, nb); break;
        case GGML_TYPE_Q8_0: dequant_q8_0(src, y, nb); break;
        case GGML_TYPE_Q3_K: dequant_q3_K(src, y, nb); break;
        case GGML_TYPE_Q4_K: dequant_q4_K(src, y, nb); break;
        case GGML_TYPE_Q5_K: dequant_q5_K(src, y, nb); break;
        case GGML_TYPE_Q6_K: dequant_q6_K(src, y, nb); break;
        default: return false;
    }
    return true;
}

bool dequantize_into(uint32_t type, const void *src, float *out, int64_t n) {
    return deq_n(type, src, out, n);
}

float *dequantize(const struct gguf_tensor *t) {
    int64_t n = 1;
    for (uint32_t i = 0; i < t->n_dims; i++) n *= (int64_t)t->dims[i];
    if (!t->data || ggml_blck_size(t->type) == 0 || n % ggml_blck_size(t->type) != 0)
        return NULL;
    float *y = malloc((size_t)n * sizeof(float));
    if (!y) return NULL;
    if (!deq_n(t->type, t->data, y, n)) { free(y); return NULL; }
    return y;
}

// Dequantize one logical row of length `row_len` starting at element row*row_len.
// Both the start element and row_len must be block-aligned. Caller frees.
float *dequantize_row(const struct gguf_tensor *t, int64_t row, int64_t row_len) {
    int blck = ggml_blck_size(t->type);
    int64_t start = row * row_len;
    if (!t->data || blck == 0 || row_len % blck != 0 || start % blck != 0) return NULL;
    const unsigned char *src = (const unsigned char *)t->data
                             + (size_t)(start / blck) * ggml_type_size(t->type);
    float *y = malloc((size_t)row_len * sizeof(float));
    if (!y) return NULL;
    if (!deq_n(t->type, src, y, row_len)) { free(y); return NULL; }
    return y;
}
