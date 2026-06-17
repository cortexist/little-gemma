// CUDA backend, int8 matmul with wide (32-bit) weight loads — "i8r" = repacked.
//
// Idea 1 (llama.cpp's mul_mat_vec_q, simplified): the activation x is the same
// for every output row, so quantize it to int8 ONCE per matmul — per 32-element
// group, a scale d_x and the int8 values (plus their sum, for weights that
// carry a min). The per-row dot is then done in INTEGER arithmetic over each
// weight sub-block, the float scale applied once per sub-block instead of once
// per element. For a weight value  w = d*sc*q (- dmin*m)  and activation
// x ≈ d_x*xq:
//     sum_i w_i x_i = d_x ( d*sc * Σ q_i·xq_i  [- dmin*m * Σ xq_i] )
// The Σ q_i·xq_i is a small-integer dot taken 4 elements at a time by __dp4a.
// This is lossy (int8 activation), exactly like llama.cpp's GPU path. It first
// shipped as a byte-load backend, model-cuda-i8.cu (in git history); idea 2
// then replaced it, bit-identical and strictly faster.
//
// Idea 2 ("r" = repacked): profiling (Nsight Compute) showed the large matmuls
// saturate the load/store unit's issue queue (lg_throttle) — every __dp4a group
// was built from 4-16 single-byte loads plus shift/or assembly. Here each group
// is ONE aligned uint32 load, and the 4 packed values are extracted with
// SIMD-in-word masks (e.g. all four q4_K nibbles at once: q32 & 0x0F0F0F0F).
// 4-16x fewer load instructions, and a warp's loads fall in a few cache sectors.
//
// Wide loads need 4-byte alignment. q4_K (144 B) and q5_K (176 B) blocks
// already have it, so they are read in place from the uploaded blob. q3_K
// (110 B) and q6_K (210 B) do not — their block stride breaks alignment — so
// those tensors are REPACKED once on the host at upload time into padded,
// 4-aligned twins (116 / 212 B, +5.5% / +1% bytes). The repack also pre-unpacks
// q3_K's twisted 6-bit scales into flat int8: the layout quirks of the file
// format are paid once per model, not per sub-block per row per token.

#include "model-cuda.cuh"

// Quantize a whole activation in one kernel (one thread per 32-element group).
// Almost every activation is quantized by its producer's epilogue instead (see
// model-cuda.cuh); this remains for the one with no producer kernel, the
// host-uploaded embedding (act_quantize).
__global__ static void quantize_act_kernel(const float *x, struct actq aq, int ng) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g < ng) d_quant_group(x + (size_t)g * 32, g, aq);
}

// ---- repacked block layouts (4-aligned strides; built on host, read on device) ----

typedef struct {                 // q3_K repacked: 110 -> 116 bytes
    ggml_half d;                 // super-block scale
    int8_t    scales[16];        // pre-unpacked from the 12-byte twist, minus 32
    uint8_t   pad[2];            // -> qs lands 4-aligned
    uint8_t   qs[64];            // low 2 bits (unchanged)
    uint8_t   hmask[32];         // high-bit mask (unchanged)
} block_q3_Kr;

typedef struct {                 // q6_K repacked: 210 -> 212 bytes (pad only)
    uint8_t   ql[128];
    uint8_t   qh[64];
    int8_t    scales[16];
    ggml_half d;
    uint8_t   pad[2];
} block_q6_Kr;

static_assert(sizeof(block_q3_Kr) == 116 && offsetof(block_q3_Kr, qs) % 4 == 0
              && offsetof(block_q3_Kr, hmask) % 4 == 0, "block_q3_Kr layout");
static_assert(sizeof(block_q6_Kr) == 212 && offsetof(block_q6_Kr, qh) % 4 == 0, "block_q6_Kr layout");


__device__ static uint32_t ld32(const void *p) { return *(const uint32_t *)p; }

// ---- one weight sub-block's contribution to the dot, in integer arithmetic ----
// The per-element index math mirrors model-cuda-f32.cu; each dp4a's weight int
// comes from one uint32 load + masks instead of 4 byte loads + shifts.

// q4_K/q5_K block strides (144/176) are multiples of 16, so their qs/qh can go
// further than uint32: one 128-bit vector load replaces four word loads.
__device__ static int nib4(uint32_t w, int half) { return (int)(half ? (w >> 4) & 0x0F0F0F0Fu : w & 0x0F0F0F0Fu); }

// d_gsm with the 12 scale bytes read as 3 words instead of 2-3 scattered bytes
// (same bits, fewer and wider loads); s must be 4-aligned (it is: offset 4).
__device__ static void d_gsm32(int j, const uint32_t *s, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = (s[0] >> (8 * j)) & 63; *m = (s[1] >> (8 * j)) & 63; }
    else {
        int b = 8 * (j - 4);
        *d = ((s[2] >> b) & 0x0F) | (((s[0] >> (b + 6)) & 3) << 4);
        *m = (((s[2] >> b) >> 4) & 0x0F) | (((s[1] >> (b + 6)) & 3) << 4);
    }
}
// d and dmin are adjacent fp16 — one word load covers both.
__device__ static void d_dm(const ggml_half *dp, float *d, float *mn) {
    uint32_t w = ld32(dp);
    *d = d_fp16((uint16_t)w); *mn = d_fp16((uint16_t)(w >> 16));
}

__device__ static float sub_q4_K(const block_q4_K *p, int sj, const int8_t *xqb, const float2 *xds) {
    uint8_t sc, mm; d_gsm32(sj, (const uint32_t *)p->scales, &sc, &mm);
    int g = sj >> 1, half = sj & 1;
    const uint4 *q = (const uint4 *)(p->qs + g * 32);
    const int4  *a = (const int4 *)(xqb + sj * 32);  // sub-block sj == activation group sj
    int dot = 0;
    for (int h = 0; h < 2; h++) {
        uint4 q4 = q[h]; int4 a4 = a[h];           // 16 weight bytes + 16 activation int8
        dot = __dp4a(nib4(q4.x, half), a4.x, dot);
        dot = __dp4a(nib4(q4.y, half), a4.y, dot);
        dot = __dp4a(nib4(q4.z, half), a4.z, dot);
        dot = __dp4a(nib4(q4.w, half), a4.w, dot);
    }
    float d, mn; d_dm(&p->d, &d, &mn);
    return xds[sj].x * (d * sc * dot - mn * mm * xds[sj].y);
}
__device__ static float sub_q5_K(const block_q5_K *p, int sj, const int8_t *xqb, const float2 *xds) {
    uint8_t sc, mm; d_gsm32(sj, (const uint32_t *)p->scales, &sc, &mm);
    int g = sj >> 1, half = sj & 1;
    const uint4 *q = (const uint4 *)(p->qs + g * 32);
    const uint4 *qh = (const uint4 *)p->qh;
    const int4  *a = (const int4 *)(xqb + sj * 32);
    int bitpos = 2 * g + half;                     // which qh bit holds this sub-block's 5th bit
    int dot = 0;                                   // 4-bit + high bit -> value 0..31
    for (int h = 0; h < 2; h++) {
        uint4 q4 = q[h], h4 = qh[h]; int4 a4 = a[h];
        dot = __dp4a(nib4(q4.x, half) | (int)(((h4.x >> bitpos) & 0x01010101u) << 4), a4.x, dot);
        dot = __dp4a(nib4(q4.y, half) | (int)(((h4.y >> bitpos) & 0x01010101u) << 4), a4.y, dot);
        dot = __dp4a(nib4(q4.z, half) | (int)(((h4.z >> bitpos) & 0x01010101u) << 4), a4.z, dot);
        dot = __dp4a(nib4(q4.w, half) | (int)(((h4.w >> bitpos) & 0x01010101u) << 4), a4.w, dot);
    }
    float d, mn; d_dm(&p->d, &d, &mn);
    return xds[sj].x * (d * sc * dot - mn * mm * xds[sj].y);
}
__device__ static float sub_q3_Kr(const block_q3_Kr *p, int sj, const int8_t *xqb, const float2 *xds) {
    int ni = sj >> 3, rem = sj & 7, j = rem >> 1, half = rem & 1;
    int shift = 2 * j, bitpos = ni * 4 + j;
    const uint8_t *qs = p->qs + ni * 32 + half * 16;
    const uint8_t *hm = p->hmask + half * 16;
    int g_act = ni * 4 + j;
    const int8_t *xqg = xqb + g_act * 32 + half * 16;
    int dot = 0;                                   // 2-bit value minus hmask offset (signed, -4..3)
    for (int l = 0; l < 16; l += 4) {
        uint32_t q32 = (ld32(qs + l) >> shift) & 0x03030303u;
        uint32_t h32 = (ld32(hm + l) >> bitpos) & 0x01010101u;
        int w = (int)__vsub4(q32, __vsub4(0x04040404u, h32 << 2));  // q - 4*(1 - hbit), per byte
        dot = __dp4a(w, *(const int *)(xqg + l), dot);
    }
    return xds[g_act].x * d_fp16(p->d) * p->scales[sj] * dot;
}
__device__ static float sub_q6_Kr(const block_q6_Kr *p, int sj, const int8_t *xqb, const float2 *xds) {
    int ni = sj >> 3, rem = sj & 7, grp = rem >> 1, hl = rem & 1;
    int sh = grp * 2, off = (grp & 1) ? 32 : 0, hin = (grp >= 2);
    const uint8_t *ql = p->ql + ni * 64 + off + hl * 16;
    const uint8_t *qh = p->qh + ni * 32 + hl * 16;
    int g_act = ni * 4 + grp;
    const int8_t *xqg = xqb + g_act * 32 + hl * 16;
    int dot = 0;                                   // 6-bit value minus 32 (signed, -32..31)
    for (int i = 0; i < 16; i += 4) {
        uint32_t l32 = ld32(ql + i);
        uint32_t lo = hin ? (l32 >> 4) & 0x0F0F0F0Fu : l32 & 0x0F0F0F0Fu;
        uint32_t hi = ((ld32(qh + i) >> sh) & 0x03030303u) << 4;
        int w = (int)__vsub4(lo | hi, 0x20202020u);
        dot = __dp4a(w, *(const int *)(xqg + i), dot);
    }
    return xds[g_act].x * d_fp16(p->d) * p->scales[ni * 8 + 2 * grp + hl] * dot;
}
__device__ static float sub_q8_0(const block_q8_0 *p, const int8_t *xqb, const float2 *xds) {
    int dot = 0;                                   // not in the test models; kept scalar
    for (int i = 0; i < 32; i++) dot += (int)p->qs[i] * xqb[i];
    return xds[0].x * d_fp16(p->d) * dot;
}

// Chunk-form siblings of the subs above: identical math in identical order
// per column — but every weight byte is loaded ONCE into registers and dotted
// against all NB activation columns. The scalar subs re-load the weight bytes
// for each column; under L1 on a discrete GPU that re-load is free, which is
// why nobody noticed — but the Orin reads its in-place (zero-copy,
// cudaHostRegister'd) q4_K/q5_K weights UNCACHED, so there the re-loads
// multiplied chunk weight traffic by NB: a B=2 MTP verify cost two full
// decode passes, and prefill chunks paid up to Bx the same way.
template <int NB>
__device__ static void sub_q4_K_n(const block_q4_K *p, int sj, const int8_t *xq0, int k,
                                  const float2 *xds0, int kg, float *s) {
    uint8_t sc, mm; d_gsm32(sj, (const uint32_t *)p->scales, &sc, &mm);
    int g = sj >> 1, half = sj & 1;
    const uint4 *q = (const uint4 *)(p->qs + g * 32);
    uint4 qa = q[0], qb = q[1];
    int w0 = nib4(qa.x, half), w1 = nib4(qa.y, half), w2 = nib4(qa.z, half), w3 = nib4(qa.w, half);
    int w4 = nib4(qb.x, half), w5 = nib4(qb.y, half), w6 = nib4(qb.z, half), w7 = nib4(qb.w, half);
    float d, mn; d_dm(&p->d, &d, &mn);
    #pragma unroll
    for (int j = 0; j < NB; j++) {
        const int4 *a = (const int4 *)(xq0 + (size_t)j * k + sj * 32);
        int4 a0 = a[0], a1 = a[1];
        int dot = 0;
        dot = __dp4a(w0, a0.x, dot); dot = __dp4a(w1, a0.y, dot);
        dot = __dp4a(w2, a0.z, dot); dot = __dp4a(w3, a0.w, dot);
        dot = __dp4a(w4, a1.x, dot); dot = __dp4a(w5, a1.y, dot);
        dot = __dp4a(w6, a1.z, dot); dot = __dp4a(w7, a1.w, dot);
        float2 xd = xds0[(size_t)j * kg + sj];
        s[j] += xd.x * (d * sc * dot - mn * mm * xd.y);
    }
}
template <int NB>
__device__ static void sub_q5_K_n(const block_q5_K *p, int sj, const int8_t *xq0, int k,
                                  const float2 *xds0, int kg, float *s) {
    uint8_t sc, mm; d_gsm32(sj, (const uint32_t *)p->scales, &sc, &mm);
    int g = sj >> 1, half = sj & 1;
    const uint4 *q = (const uint4 *)(p->qs + g * 32);
    const uint4 *qh = (const uint4 *)p->qh;
    int bitpos = 2 * g + half;
    uint4 qa = q[0], qb = q[1], ha = qh[0], hb = qh[1];
    int w0 = nib4(qa.x, half) | (int)(((ha.x >> bitpos) & 0x01010101u) << 4);
    int w1 = nib4(qa.y, half) | (int)(((ha.y >> bitpos) & 0x01010101u) << 4);
    int w2 = nib4(qa.z, half) | (int)(((ha.z >> bitpos) & 0x01010101u) << 4);
    int w3 = nib4(qa.w, half) | (int)(((ha.w >> bitpos) & 0x01010101u) << 4);
    int w4 = nib4(qb.x, half) | (int)(((hb.x >> bitpos) & 0x01010101u) << 4);
    int w5 = nib4(qb.y, half) | (int)(((hb.y >> bitpos) & 0x01010101u) << 4);
    int w6 = nib4(qb.z, half) | (int)(((hb.z >> bitpos) & 0x01010101u) << 4);
    int w7 = nib4(qb.w, half) | (int)(((hb.w >> bitpos) & 0x01010101u) << 4);
    float d, mn; d_dm(&p->d, &d, &mn);
    #pragma unroll
    for (int j = 0; j < NB; j++) {
        const int4 *a = (const int4 *)(xq0 + (size_t)j * k + sj * 32);
        int4 a0 = a[0], a1 = a[1];
        int dot = 0;
        dot = __dp4a(w0, a0.x, dot); dot = __dp4a(w1, a0.y, dot);
        dot = __dp4a(w2, a0.z, dot); dot = __dp4a(w3, a0.w, dot);
        dot = __dp4a(w4, a1.x, dot); dot = __dp4a(w5, a1.y, dot);
        dot = __dp4a(w6, a1.z, dot); dot = __dp4a(w7, a1.w, dot);
        float2 xd = xds0[(size_t)j * kg + sj];
        s[j] += xd.x * (d * sc * dot - mn * mm * xd.y);
    }
}
template <int NB>
__device__ static void sub_q3_Kr_n(const block_q3_Kr *p, int sj, const int8_t *xq0, int k,
                                   const float2 *xds0, int kg, float *s) {
    int ni = sj >> 3, rem = sj & 7, jj = rem >> 1, half = rem & 1;
    int shift = 2 * jj, bitpos = ni * 4 + jj;
    const uint8_t *qs = p->qs + ni * 32 + half * 16;
    const uint8_t *hm = p->hmask + half * 16;
    int g_act = ni * 4 + jj;
    int w[4];
    for (int l = 0; l < 16; l += 4) {
        uint32_t q32 = (ld32(qs + l) >> shift) & 0x03030303u;
        uint32_t h32 = (ld32(hm + l) >> bitpos) & 0x01010101u;
        w[l >> 2] = (int)__vsub4(q32, __vsub4(0x04040404u, h32 << 2));
    }
    float dd = d_fp16(p->d);
    float sc8 = p->scales[sj];
    #pragma unroll
    for (int j = 0; j < NB; j++) {
        const int8_t *xqg = xq0 + (size_t)j * k + g_act * 32 + half * 16;
        int dot = 0;
        for (int l = 0; l < 16; l += 4) dot = __dp4a(w[l >> 2], *(const int *)(xqg + l), dot);
        s[j] += xds0[(size_t)j * kg + g_act].x * dd * sc8 * dot;
    }
}
template <int NB>
__device__ static void sub_q6_Kr_n(const block_q6_Kr *p, int sj, const int8_t *xq0, int k,
                                   const float2 *xds0, int kg, float *s) {
    int ni = sj >> 3, rem = sj & 7, grp = rem >> 1, hl = rem & 1;
    int sh = grp * 2, off = (grp & 1) ? 32 : 0, hin = (grp >= 2);
    const uint8_t *ql = p->ql + ni * 64 + off + hl * 16;
    const uint8_t *qh = p->qh + ni * 32 + hl * 16;
    int g_act = ni * 4 + grp;
    int w[4];
    for (int i = 0; i < 16; i += 4) {
        uint32_t l32 = ld32(ql + i);
        uint32_t lo = hin ? (l32 >> 4) & 0x0F0F0F0Fu : l32 & 0x0F0F0F0Fu;
        uint32_t hi = ((ld32(qh + i) >> sh) & 0x03030303u) << 4;
        w[i >> 2] = (int)__vsub4(lo | hi, 0x20202020u);
    }
    float dd = d_fp16(p->d);
    float sc8 = p->scales[ni * 8 + 2 * grp + hl];
    #pragma unroll
    for (int j = 0; j < NB; j++) {
        const int8_t *xqg = xq0 + (size_t)j * k + g_act * 32 + hl * 16;
        int dot = 0;
        for (int i = 0; i < 16; i += 4) dot = __dp4a(w[i >> 2], *(const int *)(xqg + i), dot);
        s[j] += xds0[(size_t)j * kg + g_act].x * dd * sc8 * dot;
    }
}
template <int NB>
__device__ static void sub_q8_0_n(const block_q8_0 *p, const int8_t *xq0, int k,
                                  const float2 *xds0, int kg, float *s) {
    int8_t w[32];
    for (int i = 0; i < 32; i++) w[i] = p->qs[i];
    float dd = d_fp16(p->d);
    #pragma unroll
    for (int j = 0; j < NB; j++) {
        const int8_t *xqb = xq0 + (size_t)j * k;
        int dot = 0;
        for (int i = 0; i < 32; i++) dot += (int)w[i] * xqb[i];
        s[j] += xds0[(size_t)j * kg].x * dd * dot;
    }
}

// out[i] = W[i,:] . x : one warp per output row. For K-quant/q8_0 weights the
// lanes split the row's sub-blocks (integer dots); for bf16/f32/f16 they fall
// back to the f32 path (lane 0). A shuffle reduces the per-lane partials.
// wbase/ts refer to the repacked layout for q3_K/q6_K, the original otherwise.
// (2 rows per warp — an ILP twist on the empirically-dead split-k idea — was
// tried here and also regressed, uniform, adaptive, and template-specialized
// alike; the one-row shape stays.)
__global__ static void __launch_bounds__(256, 6)
matmul_i8r_kernel(float *out, const unsigned char *wbase, int type, int ts, int blck,
                  const float *x, const int8_t *xq, const float2 *xds, int k, int m) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= m) return;
    const unsigned char *row = wbase + (size_t)warp * (k / blck) * ts;
    int nb = k / blck, gpb = blck / 32;                 // activation groups per weight block
    float s = 0.0f;

    int nsub = 0;
    if (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) nsub = 8;
    else if (type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q6_K) nsub = 16;
    else if (type == GGML_TYPE_Q8_0) nsub = 1;

    if (nsub) {
        int total = nb * nsub;
        for (int si = lane; si < total; si += 32) {
            int b = si / nsub, sj = si % nsub;
            const unsigned char *blk = row + (size_t)b * ts;
            const int8_t *xqb = xq + (size_t)b * blck;
            const float2 *xdsb = xds + (size_t)b * gpb;
            switch (type) {
                case GGML_TYPE_Q4_K: s += sub_q4_K((const block_q4_K *)blk, sj, xqb, xdsb); break;
                case GGML_TYPE_Q5_K: s += sub_q5_K((const block_q5_K *)blk, sj, xqb, xdsb); break;
                case GGML_TYPE_Q3_K: s += sub_q3_Kr((const block_q3_Kr *)blk, sj, xqb, xdsb); break;
                case GGML_TYPE_Q6_K: s += sub_q6_Kr((const block_q6_Kr *)blk, sj, xqb, xdsb); break;
                case GGML_TYPE_Q8_0: s += sub_q8_0((const block_q8_0 *)blk, xqb, xdsb); break;
            }
        }
    } else if (type == GGML_TYPE_F32) {                  // float fallbacks: lanes split the row
        const float *wr = (const float *)row;
        for (int i = lane; i < k; i += 32) s += wr[i] * x[i];
    } else {                                             // bf16 / f16: one uint32 = 2 elements (k is even)
        const uint16_t *wr = (const uint16_t *)row;
        for (int i = 2 * lane; i < k; i += 64) {
            uint32_t two = ld32(wr + i);
            float w0 = type == GGML_TYPE_BF16 ? d_bf16((uint16_t)two) : d_fp16((uint16_t)two);
            float w1 = type == GGML_TYPE_BF16 ? d_bf16((uint16_t)(two >> 16)) : d_fp16((uint16_t)(two >> 16));
            s += w0 * x[i] + w1 * x[i + 1];
        }
    }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if (lane == 0) out[warp] = s;
}

// ---- host-side repack (once per tensor, at first use) ----------------------

// q3_K's 6-bit scale untwist (same as quant.c / the old in-kernel version),
// done once here so the device reads flat int8.
static void repack_q3_K(block_q3_Kr *dst, const block_q3_K *src, size_t nb) {
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    for (size_t i = 0; i < nb; i++) {
        uint32_t aux[4]; const int8_t *scales = (const int8_t *)aux;
        memcpy(aux, src[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = ((aux[0] >> 0) & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = ((aux[1] >> 0) & kmask2) | (((tmp >> 2) & kmask1) << 4);
        dst[i].d = src[i].d;
        for (int sj = 0; sj < 16; sj++) dst[i].scales[sj] = (int8_t)(scales[sj] - 32);
        memcpy(dst[i].qs, src[i].qs, 64);
        memcpy(dst[i].hmask, src[i].hmask, 32);
    }
}
static void repack_q6_K(block_q6_Kr *dst, const block_q6_K *src, size_t nb) {
    for (size_t i = 0; i < nb; i++) {
        memcpy(dst[i].ql, src[i].ql, 128);
        memcpy(dst[i].qh, src[i].qh, 64);
        memcpy(dst[i].scales, src[i].scales, 16);
        dst[i].d = src[i].d;
    }
}

// Per-tensor device weight table. q3_K/q6_K get a repacked copy; everything
// else aliases the blob already uploaded by ensure_weights. Filled lazily on
// first use — i.e. during the two un-captured warmup tokens, so the one-time
// cudaMallocs here are done before graph capture begins.
static unsigned char **g_rw = NULL;

static const unsigned char *rweight(const struct gguf_tensor *t, int *ts) {
    *ts = (t->type == GGML_TYPE_Q3_K) ? (int)sizeof(block_q3_Kr)
        : (t->type == GGML_TYPE_Q6_K) ? (int)sizeof(block_q6_Kr)
        : (int)ggml_type_size(t->type);
    if (!g_rw) {
        g_rw = (unsigned char **)calloc(g_ctx->header.num_tensors, sizeof *g_rw);
        if (!g_rw) { fprintf(stderr, "rweight: out of memory\n"); exit(1); }
    }
    size_t i = (size_t)(t - g_ctx->tensors);
    if (g_rw[i]) return g_rw[i];
    if (t->type != GGML_TYPE_Q3_K && t->type != GGML_TYPE_Q6_K)
        return g_rw[i] = (unsigned char *)dev_weight(t);

    int64_t n = 1;
    for (uint32_t d = 0; d < t->n_dims; d++) n *= (int64_t)t->dims[d];
    size_t nb = (size_t)(n / QK_K), bytes = nb * (size_t)*ts;
    void *host = malloc(bytes);
    if (!host) { fprintf(stderr, "rweight: out of memory repacking %s\n", t->name); exit(1); }
    if (t->type == GGML_TYPE_Q3_K) repack_q3_K((block_q3_Kr *)host, (const block_q3_K *)t->data, nb);
    else                           repack_q6_K((block_q6_Kr *)host, (const block_q6_K *)t->data, nb);
    unsigned char *dev;
    CUDA_CHECK(cudaMalloc(&dev, bytes));
    CUDA_CHECK(cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice));
    free(host);
    return g_rw[i] = dev;
}

// q6_K twins for the f32/bf16 PLE matmuls. On the Orin's E4B model the per-layer
// proj [256->2560] and inp_gate [2560->256] (and per_layer_model_proj) are stored
// F32 — the A5000's Q4_K_M conversion quantizes them to q4_K and runs them on the
// mma path, but this model keeps them full precision, so they fall to the scalar
// warp-per-row kernel: ~2% of prefill's MACs but ~11% of its time (84 tiny tensor-
// core-less matmuls on 8 SMs). Quantize them to q6_K at upload and route through
// the existing tensor-core q6_K mma path. PREFILL ONLY: matmul_q (decode) and
// matmul_q_2 (verify) keep the F32 weight, so decode stays byte-identical and
// -mtp == plain holds; only the prefilled KV shifts (relaxed class, like flash).
static unsigned char **g_rw_q6 = NULL;
static const unsigned char *rweight_q6(const struct gguf_tensor *t) {
    if (!g_rw_q6) return NULL;
    return g_rw_q6[(size_t)(t - g_ctx->tensors)];
}
// f32 row (256-multiple k) -> block_q6_K, the exact inverse of dequant_q6_K: 16
// sub-blocks of 16, a per-sub-block int8 scale over the superblock f16 d, values
// 6-bit signed (q-32) split low4=ql / high2=qh. Simple amax/32 per-sub-block scale.
static void quantize_q6_K(block_q6_K *y, const float *x, int64_t nb) {
    for (int64_t bb = 0; bb < nb; bb++, x += QK_K) {
        block_q6_K *blk = &y[bb];
        float sub[16], dmax = 0.0f;
        for (int j = 0; j < 16; j++) {
            float amax = 0.0f;
            for (int i = 0; i < 16; i++) { float a = fabsf(x[j * 16 + i]); if (a > amax) amax = a; }
            sub[j] = amax / 32.0f;
            if (sub[j] > dmax) dmax = sub[j];
        }
        float d = dmax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
        __half hd = __float2half(d);
        memcpy(&blk->d, &hd, sizeof(uint16_t));
        uint8_t q[QK_K];
        for (int j = 0; j < 16; j++) {
            int s = (int)lrintf(sub[j] * id);
            blk->scales[j] = (int8_t)(s < 0 ? 0 : s > 127 ? 127 : s);
            float eff = d * blk->scales[j], ie = eff > 0.0f ? 1.0f / eff : 0.0f;
            for (int i = 0; i < 16; i++) {
                int v = (int)lrintf(x[j * 16 + i] * ie);
                v = v < -32 ? -32 : v > 31 ? 31 : v;
                q[j * 16 + i] = (uint8_t)(v + 32);
            }
        }
        memset(blk->ql, 0, 128); memset(blk->qh, 0, 64);
        for (int g = 0; g < 2; g++) {                     // two 128-value halves
            uint8_t *ql = blk->ql + g * 64, *qh = blk->qh + g * 32;
            const uint8_t *qg = q + g * 128;
            for (int l = 0; l < 32; l++) {
                uint8_t a = qg[l], b = qg[l + 32], c = qg[l + 64], e = qg[l + 96];
                ql[l]      = (a & 0xF) | ((c & 0xF) << 4);
                ql[l + 32] = (b & 0xF) | ((e & 0xF) << 4);
                qh[l] = ((a >> 4) & 3) | (((b >> 4) & 3) << 2) | (((c >> 4) & 3) << 4) | (((e >> 4) & 3) << 6);
            }
        }
    }
}
// Build the q6_K twins up front (eager — no mid-capture malloc). LG_NO_Q6PROJ
// forces the scalar fallback (A/B); LG_Q6PROJ_VERIFY round-trips the quantizer.
static void q6_build_all(void) {
    if (getenv("LG_NO_Q6PROJ")) { fprintf(stderr, "q6_K proj repack: disabled by LG_NO_Q6PROJ (scalar fallback)\n"); return; }
    int ntensor = 0;
    for (uint64_t i = 0; i < g_ctx->header.num_tensors; i++) {
        const struct gguf_tensor *t = &g_ctx->tensors[i];
        if (t->n_dims == 2 && t->dims[0] % QK_K == 0 &&
            (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16)) ntensor++;
    }
    if (!ntensor) return;
    g_rw_q6 = (unsigned char **)calloc(g_ctx->header.num_tensors, sizeof *g_rw_q6);
    if (!g_rw_q6) return;
    int verify = getenv("LG_Q6PROJ_VERIFY") != NULL;
    size_t made = 0; int n_made = 0; double worst = 0.0;
    for (uint64_t i = 0; i < g_ctx->header.num_tensors; i++) {
        const struct gguf_tensor *t = &g_ctx->tensors[i];
        if (t->n_dims != 2 || t->dims[0] % QK_K) continue;
        if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) continue;
        int64_t k = (int64_t)t->dims[0], m = (int64_t)t->dims[1], n = k * m, nb = n / QK_K;
        float *f = (float *)malloc((size_t)n * 4);
        block_q6_K *q6 = (block_q6_K *)malloc((size_t)nb * sizeof(block_q6_K));
        block_q6_Kr *q6r = (block_q6_Kr *)malloc((size_t)nb * sizeof(block_q6_Kr));
        if (!f || !q6 || !q6r || !dequantize_into(t->type, t->data, f, n)) { free(f); free(q6); free(q6r); continue; }
        quantize_q6_K(q6, f, nb);
        if (verify) {                                     // round-trip vs the original f32
            float *chk = (float *)malloc((size_t)n * 4);
            if (chk && dequantize_into(GGML_TYPE_Q6_K, q6, chk, n)) {
                double num = 0, den = 0;
                for (int64_t e = 0; e < n; e++) { double dd = (double)chk[e] - f[e]; num += dd * dd; den += (double)f[e] * f[e]; }
                double rel = den > 0 ? sqrt(num / den) : 0;
                if (rel > worst) worst = rel;
            }
            free(chk);
        }
        repack_q6_K(q6r, q6, nb);
        free(f); free(q6);
        size_t bytes = (size_t)nb * sizeof(block_q6_Kr);
        unsigned char *dev;
        if (cudaMalloc(&dev, bytes) != cudaSuccess) { cudaGetLastError(); free(q6r); continue; }
        cudaMemcpy(dev, q6r, bytes, cudaMemcpyHostToDevice);
        free(q6r);
        g_rw_q6[(size_t)i] = dev; made += bytes; n_made++;
    }
    fprintf(stderr, "q6_K proj repack: %d tensor(s), %.1f MB -> mma prefill path", n_made, made / 1e6);
    if (verify) fprintf(stderr, " (worst round-trip rel-rmse %.2e)", worst);
    fprintf(stderr, "\n");
}

// Quantized activation scratch (grows as needed; reused across matmuls).
static int8_t *g_xq = NULL; static float2 *g_xds = NULL;
static int g_act_cap = 0;
static void ensure_act(int k) {
    if (k <= g_act_cap) return;
    cudaFree(g_xq); cudaFree(g_xds);
    CUDA_CHECK(cudaMalloc(&g_xq, (size_t)k));
    CUDA_CHECK(cudaMalloc(&g_xds, (size_t)(k / 32) * sizeof(float2)));
    g_act_cap = k;
}

// The backend seam. The activation's int8 form is already in g_xq/g_xds —
// put there by the producer kernel's epilogue (actq_for) or by act_quantize.
// (d_x is still needed for the bf16/f32/f16 fallback path, which dots floats.)
static struct actq actq_for(int k) {
    ensure_act(k);
    struct actq aq = { g_xq, g_xds };
    return aq;
}
static void act_quantize(const float *d_x, int k) {
    int ng = k / 32;
    quantize_act_kernel<<<gridn(ng), 256>>>(d_x, actq_for(k), ng);
}
// Repack/upload every quantized tensor up front: lazy per-tensor uploads would
// otherwise happen between fork events, unordered against the side stream.
static void rweight_init_all(void) {
    static int done = 0;
    if (done) return;
    for (uint64_t i = 0; i < g_ctx->header.num_tensors; i++) {
        const struct gguf_tensor *t = &g_ctx->tensors[i];
        int ts;
        if (ggml_blck_size(t->type) == QK_K || t->type == GGML_TYPE_Q8_0) rweight(t, &ts);
    }
    q6_build_all();
    CUDA_CHECK(cudaDeviceSynchronize());
    done = 1;
}

static void matmul_q(float *d_out, const struct gguf_tensor *t, const float *d_x, int k, int m) {
    rweight_init_all();
    int blck = ggml_blck_size(t->type), ts;
    const unsigned char *w = rweight(t, &ts);
    int rows_per_block = 256 / 32;
    int blocks = (m + rows_per_block - 1) / rows_per_block;
    matmul_i8r_kernel<<<blocks, 256, 0, g_launch>>>(d_out, w, (int)t->type, ts, blck, d_x, g_xq, g_xds, k, m);
}

// The chunk form: one warp per output row, all PREFILL_B activation columns at
// once — out[j*m + i] = W[i,:] . x_j. This is where batched prefill's win
// lives: a weight sub-block is fetched and dotted against every column while
// it is hot, so the row crosses DRAM once per CHUNK instead of once per token
// (the per-column re-reads hit L1). Activations sit at column stride k (int8)
// and k/32 (scales), exactly as the producers' epilogues laid them out. Each
// column's accumulation order matches the one-column kernel, so the result is
// bit-identical to NB separate matmul_q calls — an invariant the <2>
// instantiation (the MTP verify pair) MUST keep: spec-vs-plain byte-identity
// rests on the verify's argmax matching decode's exactly. The <PREFILL_B>
// instantiation relaxes it in the float fallbacks only (wide loads
// reassociate; same gate class as the mma kernels). NB is a template
// parameter so the register accumulators stay compile-sized — separate
// instruction streams, like d_attn's RING.
template <int NB>
__global__ static void matmul_i8r_n_kernel(float *out, const unsigned char *wbase, int type, int ts, int blck,
                                           const float *x, const int8_t *xq, const float2 *xds, int k, int m) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= m) return;
    const unsigned char *row = wbase + (size_t)warp * (k / blck) * ts;
    int nb = k / blck, gpb = blck / 32;
    float s[NB];
    #pragma unroll
    for (int j = 0; j < NB; j++) s[j] = 0.0f;

    int nsub = 0;
    if (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) nsub = 8;
    else if (type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q6_K) nsub = 16;
    else if (type == GGML_TYPE_Q8_0) nsub = 1;

    if (nsub) {
        int total = nb * nsub;
        for (int si = lane; si < total; si += 32) {
            int b = si / nsub, sj = si % nsub;
            const unsigned char *blk = row + (size_t)b * ts;
            const int8_t *xq0 = xq + (size_t)b * blck;       // column 0; stride k per column
            const float2 *xds0 = xds + (size_t)b * gpb;      // column 0; stride k/32
            switch (type) {
                case GGML_TYPE_Q4_K: sub_q4_K_n<NB>((const block_q4_K *)blk, sj, xq0, k, xds0, k / 32, s); break;
                case GGML_TYPE_Q5_K: sub_q5_K_n<NB>((const block_q5_K *)blk, sj, xq0, k, xds0, k / 32, s); break;
                case GGML_TYPE_Q3_K: sub_q3_Kr_n<NB>((const block_q3_Kr *)blk, sj, xq0, k, xds0, k / 32, s); break;
                case GGML_TYPE_Q6_K: sub_q6_Kr_n<NB>((const block_q6_Kr *)blk, sj, xq0, k, xds0, k / 32, s); break;
                case GGML_TYPE_Q8_0: sub_q8_0_n<NB>((const block_q8_0 *)blk, xq0, k, xds0, k / 32, s); break;
            }
        }
    } else if (type == GGML_TYPE_F32) {
        // These are the tiny-m float matmuls (laurel, altup): a handful of
        // CTAs, long serial load chains — issue-bound, ~85 launches per chunk
        // at 10% of prefill GPU time for 2% of its MACs. (A cached device
        // copy of the weights was tried first and measured NOTHING: the x
        // loads outnumber the weight load 16:1, so weight caching was never
        // the constraint. Wider loads on both sides are.)
        const float *wr = (const float *)row;
        if (NB == LG_MTP_N || (k & 3)) {                 // verify (B=LG_MTP_N): decode's order, exactly
            for (int i = lane; i < k; i += 32) {
                float w = wr[i];
                for (int j = 0; j < NB; j++) s[j] += w * x[(size_t)j * k + i];
            }
        } else {
            for (int i = 4 * lane; i < k; i += 128) {
                float4 w4 = *(const float4 *)(wr + i);
                for (int j = 0; j < NB; j++) {
                    float4 x4 = *(const float4 *)(x + (size_t)j * k + i);
                    s[j] += w4.x * x4.x + w4.y * x4.y + w4.z * x4.z + w4.w * x4.w;
                }
            }
        }
    } else {                                             // bf16 / f16
        const uint16_t *wr = (const uint16_t *)row;
        if (NB == LG_MTP_N || (k & 7)) {                 // one uint32 = 2 elements (verify B=LG_MTP_N: decode order)
            for (int i = 2 * lane; i < k; i += 64) {
                uint32_t two = ld32(wr + i);
                float w0 = type == GGML_TYPE_BF16 ? d_bf16((uint16_t)two) : d_fp16((uint16_t)two);
                float w1 = type == GGML_TYPE_BF16 ? d_bf16((uint16_t)(two >> 16)) : d_fp16((uint16_t)(two >> 16));
                for (int j = 0; j < NB; j++)
                    s[j] += w0 * x[(size_t)j * k + i] + w1 * x[(size_t)j * k + i + 1];
            }
        } else {                                         // prefill: one uint4 = 8 elements
            for (int i = 8 * lane; i < k; i += 256) {
                uint4 w8 = *(const uint4 *)(wr + i);
                uint32_t ww[4] = { w8.x, w8.y, w8.z, w8.w };
                float w[8];
                #pragma unroll
                for (int h = 0; h < 4; h++) {
                    w[2 * h]     = type == GGML_TYPE_BF16 ? d_bf16((uint16_t)ww[h]) : d_fp16((uint16_t)ww[h]);
                    w[2 * h + 1] = type == GGML_TYPE_BF16 ? d_bf16((uint16_t)(ww[h] >> 16)) : d_fp16((uint16_t)(ww[h] >> 16));
                }
                for (int j = 0; j < NB; j++) {
                    const float *xj = x + (size_t)j * k + i;
                    float4 xa = *(const float4 *)xj, xb = *(const float4 *)(xj + 4);
                    s[j] += w[0] * xa.x + w[1] * xa.y + w[2] * xa.z + w[3] * xa.w
                          + w[4] * xb.x + w[5] * xb.y + w[6] * xb.z + w[7] * xb.w;
                }
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < NB; j++) {
        float v = s[j];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) out[(size_t)j * m + warp] = v;
    }
}

// ==== the tensor-core chunk matmul (q4_K, prefill only) ======================
// The dp4a chunk kernel above is issue-bound: every 4-element dot pays its own
// unpack ALU chain, one warp per row, ~1 int8-TOPS on the Orin (the journal's
// prefill chapter has the full audit trail — B-sweeps and memory staging were
// measured dead ends). Here one mma.m16n8k32 instruction does 4,096 MACs, so
// a warp covers 16 rows x 16 columns and each unpacked weight nibble feeds 16
// columns through one instruction instead of 16 dp4a chains.
//
// Q4_K only (85% of chunk MACs on both Q4_K_M models — the coverage probe
// below); everything else stays on the dp4a kernel. PREFILL ONLY: decode and
// the B=2 MTP verify never run this, so their byte-identity gates are
// untouched. The integer sub-block dots here are EXACT (s32 accumulate, same
// values dp4a produces); only the float ORDER of combining sub-block
// contributions differs — each (row, column) is summed by one lane in
// ascending k — so chunked prefill is no longer byte-identical to per-token
// prefill, the same gate relaxation the online-softmax step shipped with
// (deterministic + numerically equivalent; the step-2 harness measured this
// kernel slightly CLOSER to a double reference than the dp4a path).
// LG_NO_MMA=1 forces the dp4a path back for A/B.
//
// Per CTA: 8 warps, 128 rows x 16 columns. Per q4_K block (256 weights), the
// CTA stages all 16 columns' activations once (two barriers per block, ten
// per typical k); per sub-block, each warp stages its 16 rows' unpacked
// nibbles warp-synchronously (no CTA barrier), builds fragments by direct
// shared reads (the lane mapping is the documented m16n8k32 layout — see
// .scratch/mma_test.cu, the validating harness), and shuffle-broadcasts the
// per-row scale pairs from the 16 lanes that hold block headers in registers.

__device__ static void d_gsm32r(int j, uint32_t s0, uint32_t s1, uint32_t s2, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = (s0 >> (8 * j)) & 63; *m = (s1 >> (8 * j)) & 63; }
    else {
        int b = 8 * (j - 4);
        *d = ((s2 >> b) & 0x0F) | (((s0 >> (b + 6)) & 3) << 4);
        *m = (((s2 >> b) >> 4) & 0x0F) | (((s1 >> (b + 6)) & 3) << 4);
    }
}

// One thread's share of staging activation block `blk` into buffer `buf`,
// issued as cp.async so it overlaps the previous block's mma work.
// The staged activation columns are 256 B each, and 256/4 = 64 words is a
// multiple of the 32 shared banks — so consecutive columns alias the same
// banks and a warp's fragment read (one column per gid) serialised ~8 ways
// (ncu: 5.8 wavefronts/load, 78% of them conflict overhead). Padding the
// per-column stride to 272 B (68 words; 68 mod 32 = 4) makes gid*4 + tid span
// all 32 banks bijectively -> conflict-free. Pure storage relocation: values
// and mma order are unchanged, so the output stays byte-identical.
#define SB_COL 272
// The A (weight nibble) tile rows are 32 B = 8 words, also a 32-bank multiple,
// so the fragment read (one row per gid) aliases gid with gid+4 — a 2-way
// conflict (the ~1.43M left after the B-side fix). Padding the A row stride to
// 48 B (12 words; gid*12 mod 32 over gid 0..7 hits 8 distinct bank-quads) makes
// gid*12 + tid span all 32 banks. q4_K only; q6_K's 16 B rows already spread.
#define SA_ROW 48
// The per-column scale pairs (sBxds) are float2 — LDS.64 reads. 8 float2/col
// = 32 words is again a 32-bank multiple, so the four tid-groups (columns 2
// apart) alias one bank: a 4-way conflict, and ncu pinned it as the entire
// ~1.23M residual left after the A/B fixes. Pad to 9 float2/col (36 words;
// tid*4 hits 4 distinct banks) -> conflict-free. Same for q4_K and q6_K.
#define SBX 9
template<int COLS>
__device__ static void mma_stage_b(int8_t *sB, float2 *sBxds, int buf,
                                   const int8_t *xq, const float2 *xds, int k, int blk, int tix) {
    // 256 threads stage COLS activation columns: each column is 256 B = 16
    // pieces of 16 B (COLS*16 cp.async total), and COLS*8 sub-block (scale,sum)
    // float2 land with plain stores (it's the buffer nobody reads until the
    // barrier, so the double-buffer safety holds either way).
    for (int t = tix; t < COLS * 16; t += blockDim.x) {
        int col = t >> 4, off = (t & 15) * 16;
        unsigned dst = (unsigned)__cvta_generic_to_shared(sB + buf * COLS * SB_COL + col * SB_COL + off);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(xq + (size_t)col * k + blk * 256 + off));
    }
    for (int t = tix; t < COLS * 8; t += blockDim.x) {
        int xcol = t >> 3, g = t & 7;
        sBxds[buf * COLS * SBX + xcol * SBX + g] = xds[(size_t)xcol * (k / 32) + blk * 8 + g];
    }
}

// ==== fat-tile MMQ q4_K kernel (the q4_K prefill matmul; in-place decode) =====
// llama.cpp's measured regime, which beats us on the Orin at HALF our occupancy:
// a wide COLS-column tile (COLS = PREFILL_B = 128) so each staged weight feeds
// 128 mma-columns — 4x the 32-wide chunk this replaced — and each warp owns ONE
// 16-row tile across all COLS cols, so the accumulator is acc[COLS/8][4] = 64
// floats/thread = 64 independent mma chains whose ILP hides the shared-load
// latency that warp-count never did. Runs ~1 CTA/SM at ~254 regs (carveout +
// launch_bounds(256,1)); the Orin ncu confirmed occ 16.5% == llama.cpp's 16.66%.
// Reads q4_K IN PLACE: a coalesced second-copy twin was measured PURE HARM on the
// Orin (its 2.33 GB footprint cost more L2/bandwidth than the coalescing saved —
// in-place 227 vs twin 198 tok/s), so the twin was deleted. Eight warps tile 128
// rows; decode/mma/scale math matches the per-token path, so output is the
// deterministic relaxed-float-order class (gated == dp4a, -mtp == plain).
template<int COLS>
__global__ static void __launch_bounds__(256, 1)
matmul_q4k_mmq_kernel(float *out, const unsigned char *wbase, int ts,
                      const int8_t *xq, const float2 *xds, int k, int m) {
    extern __shared__ unsigned char sh[];
    int8_t *sB    = (int8_t *)sh;                                  // [2][COLS][SB_COL]
    float2 *sBxds = (float2 *)(sh + 2 * COLS * SB_COL);            // [2][COLS][SBX]
    int8_t *sA    = (int8_t *)(sh + 2 * COLS * SB_COL + 2 * COLS * SBX * 8); // [wpc][2 halves][16][SA_ROW]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wpc = blockDim.x >> 5;
    const int r0 = blockIdx.x * (16 * wpc) + warp * 16;            // this warp's 16-row tile
    const int gid = lane >> 2, tid = lane & 3;
    const int nbk = k / 256;
    // gridDim.y column-tiling (wide chunk): this block owns column tile blockIdx.y,
    // i.e. the COLS columns at offset blockIdx.y*COLS. blockIdx.y==0 (every legacy
    // single-tile launch) => zero offset => byte-identical to the narrow path.
    xq  += (size_t)blockIdx.y * COLS * k;
    xds += (size_t)blockIdx.y * COLS * (k / 32);
    out += (size_t)blockIdx.y * COLS * m;
    int8_t *sAw = sA + warp * 2 * 16 * SA_ROW;
    float acc[COLS / 8][4];
    #pragma unroll
    for (int h = 0; h < COLS / 8; h++) for (int s = 0; s < 4; s++) acc[h][s] = 0.0f;

    int rowL = r0 + (lane & 15); if (rowL >= m) rowL = m - 1;      // lanes 0..15 -> this warp's 16 rows
    const unsigned char *myrow = wbase + (size_t)rowL * nbk * ts;

    mma_stage_b<COLS>(sB, sBxds, 0, xq, xds, k, 0, threadIdx.x);   // prologue: block 0 in flight
    asm volatile("cp.async.commit_group;\n");

    for (int blk = 0; blk < nbk; blk++) {
        int buf = blk & 1;
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        if (blk + 1 < nbk) {
            mma_stage_b<COLS>(sB, sBxds, buf ^ 1, xq, xds, k, blk + 1, threadIdx.x);
            asm volatile("cp.async.commit_group;\n");
        }
        const int8_t *sBb = sB + buf * COLS * SB_COL;
        const float2 *sBxd = sBxds + buf * COLS * SBX;

        const block_q4_K *hp = (const block_q4_K *)(myrow + (size_t)blk * ts);
        uint32_t s0 = ((const uint32_t *)hp->scales)[0];
        uint32_t s1 = ((const uint32_t *)hp->scales)[1];
        uint32_t s2 = ((const uint32_t *)hp->scales)[2];
        float dD, dM; d_dm(&hp->d, &dD, &dM);

        for (int sjp = 0; sjp < 4; sjp++) {
            {
                int ar = lane & 15, hi = lane >> 4;                // row-in-tile, which 16B half
                int row = r0 + ar; if (row >= m) row = m - 1;
                const block_q4_K *bp = (const block_q4_K *)(wbase + (size_t)row * nbk * ts) + blk;
                uint4 q4 = *(const uint4 *)(bp->qs + sjp * 32 + hi * 16);
                uint32_t *lo = (uint32_t *)(sAw + ar * SA_ROW + hi * 16);
                uint32_t *hh = (uint32_t *)(sAw + 16 * SA_ROW + ar * SA_ROW + hi * 16);
                lo[0] = (uint32_t)nib4(q4.x, 0); lo[1] = (uint32_t)nib4(q4.y, 0);
                lo[2] = (uint32_t)nib4(q4.z, 0); lo[3] = (uint32_t)nib4(q4.w, 0);
                hh[0] = (uint32_t)nib4(q4.x, 1); hh[1] = (uint32_t)nib4(q4.y, 1);
                hh[2] = (uint32_t)nib4(q4.z, 1); hh[3] = (uint32_t)nib4(q4.w, 1);
            }
            __syncwarp();
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                int sj = sjp * 2 + half;
                uint8_t sc, mm; d_gsm32r(sj, s0, s1, s2, &sc, &mm);
                float dscL = dD * sc, mnmL = dM * mm;              // my header ROW's pair
                float dsc0 = __shfl_sync(0xffffffffu, dscL, gid),     dsc1 = __shfl_sync(0xffffffffu, dscL, gid + 8);
                float mnm0 = __shfl_sync(0xffffffffu, mnmL, gid),     mnm1 = __shfl_sync(0xffffffffu, mnmL, gid + 8);
                const int8_t *sAh = sAw + half * 16 * SA_ROW;
                uint32_t a0 = *(uint32_t *)(sAh + gid * SA_ROW + tid * 4);
                uint32_t a1 = *(uint32_t *)(sAh + (gid + 8) * SA_ROW + tid * 4);
                uint32_t a2 = *(uint32_t *)(sAh + gid * SA_ROW + tid * 4 + 16);
                uint32_t a3 = *(uint32_t *)(sAh + (gid + 8) * SA_ROW + tid * 4 + 16);
                #pragma unroll
                for (int h = 0; h < COLS / 8; h++) {
                    int colb = h * 8 + gid;
                    uint32_t b0 = *(uint32_t *)(sBb + colb * SB_COL + sj * 32 + tid * 4);
                    uint32_t b1 = *(uint32_t *)(sBb + colb * SB_COL + sj * 32 + tid * 4 + 16);
                    float2 xd0 = sBxd[(h * 8 + tid * 2) * SBX + sj];
                    float2 xd1 = sBxd[(h * 8 + tid * 2 + 1) * SBX + sj];
                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                    acc[h][0] += xd0.x * (dsc0 * (float)c0 - mnm0 * xd0.y);
                    acc[h][1] += xd1.x * (dsc0 * (float)c1 - mnm0 * xd1.y);
                    acc[h][2] += xd0.x * (dsc1 * (float)c2 - mnm1 * xd0.y);
                    acc[h][3] += xd1.x * (dsc1 * (float)c3 - mnm1 * xd1.y);
                }
            }
            __syncwarp();
        }
    }

    #pragma unroll
    for (int h = 0; h < COLS / 8; h++) {
        int col0 = h * 8 + tid * 2;
        int rA = r0 + gid, rB = r0 + gid + 8;
        if (rA < m) { out[(size_t)col0 * m + rA] = acc[h][0]; out[(size_t)(col0 + 1) * m + rA] = acc[h][1]; }
        if (rB < m) { out[(size_t)col0 * m + rB] = acc[h][2]; out[(size_t)(col0 + 1) * m + rB] = acc[h][3]; }
    }
}

// ==== q4_K MMQ, 2 row-minitiles per warp (LG_Q4K_2TILE) =====================
// ncu on the fat 1-tile kernel: the big FFN is L1/shared-THROUGHPUT bound (L1
// 56%, SM only 44%, ALU 24%) — it re-reads the activation B-fragment + its
// per-column scale from shared on EVERY mma. llama.cpp's mul_mat_q hits 58% SM
// at the same occupancy because its 64 accumulators are arranged as ntx=2 row
// minitiles x 32 cols, so each B+scale shared load feeds TWO mmas (the two row
// tiles), halving per-mma shared traffic. This kernel does the same: a warp owns
// TWO 16-row tiles (32 rows), COLS=64 cols -> acc[2][COLS/8][4] = 64 (same regs/
// ILP as the 1-tile COLS=128), and the inner loop loads B+scale ONCE and issues
// two mmas. Math/scale path identical to the 1-tile kernel (same nib4/d_gsm32r/
// d_dm/combine) -> same relaxed-float-order class. Eight warps tile 256 rows.
// One (256-row x COLS) output tile, two 16-row minitiles per warp. xq/xds/out are
// already offset to this tile's column block; r0 is this warp's row base; sAw is
// this warp's A region. Factored out so both the conventional grid (one tile per
// CTA) and the persistent stream-K grid (one CTA grinds many tiles) reuse it.
template<int COLS>
__device__ __forceinline__ void q4k_2tile_tile(
        float *out, const unsigned char *wbase, int ts,
        const int8_t *xq, const float2 *xds, int k, int m,
        int8_t *sB, float2 *sBxds, int8_t *sAw, int r0) {
    const int lane = threadIdx.x & 31;
    const int gid = lane >> 2, tid = lane & 3;
    const int nbk = k / 256;
    float acc[2][COLS / 8][4];
    #pragma unroll
    for (int t = 0; t < 2; t++) for (int h = 0; h < COLS / 8; h++) for (int s = 0; s < 4; s++) acc[t][h][s] = 0.0f;

    int rowL0 = r0 + (lane & 15);          if (rowL0 >= m) rowL0 = m - 1;   // tile 0 header row
    int rowL1 = r0 + 16 + (lane & 15);     if (rowL1 >= m) rowL1 = m - 1;   // tile 1 header row
    const unsigned char *myrow0 = wbase + (size_t)rowL0 * nbk * ts;
    const unsigned char *myrow1 = wbase + (size_t)rowL1 * nbk * ts;

    mma_stage_b<COLS>(sB, sBxds, 0, xq, xds, k, 0, threadIdx.x);
    asm volatile("cp.async.commit_group;\n");

    for (int blk = 0; blk < nbk; blk++) {
        int buf = blk & 1;
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        if (blk + 1 < nbk) {
            mma_stage_b<COLS>(sB, sBxds, buf ^ 1, xq, xds, k, blk + 1, threadIdx.x);
            asm volatile("cp.async.commit_group;\n");
        }
        const int8_t *sBb = sB + buf * COLS * SB_COL;
        const float2 *sBxd = sBxds + buf * COLS * SBX;

        const block_q4_K *hp0 = (const block_q4_K *)(myrow0 + (size_t)blk * ts);
        const block_q4_K *hp1 = (const block_q4_K *)(myrow1 + (size_t)blk * ts);
        uint32_t s00 = ((const uint32_t *)hp0->scales)[0], s10 = ((const uint32_t *)hp0->scales)[1], s20 = ((const uint32_t *)hp0->scales)[2];
        uint32_t s01 = ((const uint32_t *)hp1->scales)[0], s11 = ((const uint32_t *)hp1->scales)[1], s21 = ((const uint32_t *)hp1->scales)[2];
        float dD0, dM0, dD1, dM1; d_dm(&hp0->d, &dD0, &dM0); d_dm(&hp1->d, &dD1, &dM1);

        for (int sjp = 0; sjp < 4; sjp++) {
            #pragma unroll
            for (int t = 0; t < 2; t++) {                          // stage both tiles' nibbles
                int ar = lane & 15, hi = lane >> 4;
                int row = r0 + t * 16 + ar; if (row >= m) row = m - 1;
                const block_q4_K *bp = (const block_q4_K *)(wbase + (size_t)row * nbk * ts) + blk;
                uint4 q4 = *(const uint4 *)(bp->qs + sjp * 32 + hi * 16);
                int8_t *sAt = sAw + t * 2 * 16 * SA_ROW;
                uint32_t *lo = (uint32_t *)(sAt + ar * SA_ROW + hi * 16);
                uint32_t *hh = (uint32_t *)(sAt + 16 * SA_ROW + ar * SA_ROW + hi * 16);
                lo[0] = (uint32_t)nib4(q4.x, 0); lo[1] = (uint32_t)nib4(q4.y, 0);
                lo[2] = (uint32_t)nib4(q4.z, 0); lo[3] = (uint32_t)nib4(q4.w, 0);
                hh[0] = (uint32_t)nib4(q4.x, 1); hh[1] = (uint32_t)nib4(q4.y, 1);
                hh[2] = (uint32_t)nib4(q4.z, 1); hh[3] = (uint32_t)nib4(q4.w, 1);
            }
            __syncwarp();
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                int sj = sjp * 2 + half;
                uint8_t sc0, mm0, sc1, mm1;
                d_gsm32r(sj, s00, s10, s20, &sc0, &mm0);
                d_gsm32r(sj, s01, s11, s21, &sc1, &mm1);
                float dscL0 = dD0 * sc0, mnmL0 = dM0 * mm0, dscL1 = dD1 * sc1, mnmL1 = dM1 * mm1;
                float dsc0_0 = __shfl_sync(0xffffffffu, dscL0, gid), dsc1_0 = __shfl_sync(0xffffffffu, dscL0, gid + 8);
                float mnm0_0 = __shfl_sync(0xffffffffu, mnmL0, gid), mnm1_0 = __shfl_sync(0xffffffffu, mnmL0, gid + 8);
                float dsc0_1 = __shfl_sync(0xffffffffu, dscL1, gid), dsc1_1 = __shfl_sync(0xffffffffu, dscL1, gid + 8);
                float mnm0_1 = __shfl_sync(0xffffffffu, mnmL1, gid), mnm1_1 = __shfl_sync(0xffffffffu, mnmL1, gid + 8);
                const int8_t *sAh0 = sAw + 0 * 2 * 16 * SA_ROW + half * 16 * SA_ROW;
                const int8_t *sAh1 = sAw + 1 * 2 * 16 * SA_ROW + half * 16 * SA_ROW;
                uint32_t a0_0 = *(uint32_t *)(sAh0 + gid * SA_ROW + tid * 4), a1_0 = *(uint32_t *)(sAh0 + (gid + 8) * SA_ROW + tid * 4);
                uint32_t a2_0 = *(uint32_t *)(sAh0 + gid * SA_ROW + tid * 4 + 16), a3_0 = *(uint32_t *)(sAh0 + (gid + 8) * SA_ROW + tid * 4 + 16);
                uint32_t a0_1 = *(uint32_t *)(sAh1 + gid * SA_ROW + tid * 4), a1_1 = *(uint32_t *)(sAh1 + (gid + 8) * SA_ROW + tid * 4);
                uint32_t a2_1 = *(uint32_t *)(sAh1 + gid * SA_ROW + tid * 4 + 16), a3_1 = *(uint32_t *)(sAh1 + (gid + 8) * SA_ROW + tid * 4 + 16);
                #pragma unroll
                for (int h = 0; h < COLS / 8; h++) {
                    int colb = h * 8 + gid;
                    uint32_t b0 = *(uint32_t *)(sBb + colb * SB_COL + sj * 32 + tid * 4);   // ONE B load...
                    uint32_t b1 = *(uint32_t *)(sBb + colb * SB_COL + sj * 32 + tid * 4 + 16);
                    float2 xd0 = sBxd[(h * 8 + tid * 2) * SBX + sj];                        // ...ONE scale load...
                    float2 xd1 = sBxd[(h * 8 + tid * 2 + 1) * SBX + sj];
                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0, e0 = 0, e1 = 0, e2 = 0, e3 = 0;
                    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3) : "r"(a0_0), "r"(a1_0), "r"(a2_0), "r"(a3_0), "r"(b0), "r"(b1));  // ...TWO mmas
                    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+r"(e0), "+r"(e1), "+r"(e2), "+r"(e3) : "r"(a0_1), "r"(a1_1), "r"(a2_1), "r"(a3_1), "r"(b0), "r"(b1));
                    acc[0][h][0] += xd0.x * (dsc0_0 * (float)c0 - mnm0_0 * xd0.y);
                    acc[0][h][1] += xd1.x * (dsc0_0 * (float)c1 - mnm0_0 * xd1.y);
                    acc[0][h][2] += xd0.x * (dsc1_0 * (float)c2 - mnm1_0 * xd0.y);
                    acc[0][h][3] += xd1.x * (dsc1_0 * (float)c3 - mnm1_0 * xd1.y);
                    acc[1][h][0] += xd0.x * (dsc0_1 * (float)e0 - mnm0_1 * xd0.y);
                    acc[1][h][1] += xd1.x * (dsc0_1 * (float)e1 - mnm0_1 * xd1.y);
                    acc[1][h][2] += xd0.x * (dsc1_1 * (float)e2 - mnm1_1 * xd0.y);
                    acc[1][h][3] += xd1.x * (dsc1_1 * (float)e3 - mnm1_1 * xd1.y);
                }
            }
            __syncwarp();
        }
    }
    #pragma unroll
    for (int t = 0; t < 2; t++)
        #pragma unroll
        for (int h = 0; h < COLS / 8; h++) {
            int col0 = h * 8 + tid * 2;
            int rA = r0 + t * 16 + gid, rB = r0 + t * 16 + gid + 8;
            if (rA < m) { out[(size_t)col0 * m + rA] = acc[t][h][0]; out[(size_t)(col0 + 1) * m + rA] = acc[t][h][1]; }
            if (rB < m) { out[(size_t)col0 * m + rB] = acc[t][h][2]; out[(size_t)(col0 + 1) * m + rB] = acc[t][h][3]; }
        }
    __syncthreads();    // a persistent CTA must finish reading sB before the next tile's prologue overwrites it
}

template<int COLS>
__global__ static void __launch_bounds__(256, 1)
matmul_q4k_mmq2_kernel(float *out, const unsigned char *wbase, int ts,
                       const int8_t *xq, const float2 *xds, int k, int m) {
    extern __shared__ unsigned char sh[];
    int8_t *sB    = (int8_t *)sh;
    float2 *sBxds = (float2 *)(sh + 2 * COLS * SB_COL);
    int8_t *sA    = (int8_t *)(sh + 2 * COLS * SB_COL + 2 * COLS * SBX * 8);
    const int warp = threadIdx.x >> 5, wpc = blockDim.x >> 5;
    q4k_2tile_tile<COLS>(out + (size_t)blockIdx.y * COLS * m, wbase, ts,
                         xq + (size_t)blockIdx.y * COLS * k, xds + (size_t)blockIdx.y * COLS * (k / 32),
                         k, m, sB, sBxds, sA + warp * 2 * 2 * 16 * SA_ROW, blockIdx.x * (32 * wpc) + warp * 32);
}

// Stream-K stage 1: persistent CTAs (grid = nsm). Each CTA grinds a strided slice
// of the (row-tile x col-tile) work-list whole-tile at a time — no K-split/fixup
// yet. Tests whether CTA-residency (8 CTAs x ~10 tiles vs 80 CTAs in 10 waves) is
// the 44->58% SM lever the ncu pinned (llama.cpp runs grid=nsm here).
template<int COLS>
__global__ static void __launch_bounds__(256, 1)
matmul_q4k_sk_kernel(float *out, const unsigned char *wbase, int ts,
                     const int8_t *xq, const float2 *xds, int k, int m, int ntx) {
    extern __shared__ unsigned char sh[];
    int8_t *sB    = (int8_t *)sh;
    float2 *sBxds = (float2 *)(sh + 2 * COLS * SB_COL);
    int8_t *sA    = (int8_t *)(sh + 2 * COLS * SB_COL + 2 * COLS * SBX * 8);
    const int warp = threadIdx.x >> 5, wpc = blockDim.x >> 5;
    int8_t *sAw = sA + warp * 2 * 2 * 16 * SA_ROW;
    const int rows_per_tile = 32 * wpc;
    const int nty = (m + rows_per_tile - 1) / rows_per_tile;
    const int ntiles = nty * ntx;
    for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        int rt = tile / ntx, ct = tile % ntx;
        q4k_2tile_tile<COLS>(out + (size_t)ct * COLS * m, wbase, ts,
                             xq + (size_t)ct * COLS * k, xds + (size_t)ct * COLS * (k / 32),
                             k, m, sB, sBxds, sAw, rt * rows_per_tile + warp * 32);
    }
}

// ==== the q6_K twin (mma.m16n8k16) ==========================================
// Same CTA shape and double-buffered activation staging as the q4_K kernel;
// what changes is the A side. q6_K carries a per-16-element int8 scale, so the
// k16 mma covers exactly one sub-block per instruction — exact integer dot,
// one float scale after, and no min term at all (6-bit values are signed).
// Validated the same way: .scratch/mma6_test.cu pins the m16n8k16 fragment
// mapping (A 2 regs row gid/gid+8, B 1 reg col gid, C as in k32) against a
// double reference before this kernel existed.
//
// Two layout notes earned by inspection rather than the profiler:
// - block_q6_Kr is 212 B: 4-aligned but NOT 16-aligned, so staging reads are
//   ld32 (like the dp4a path), never uint4.
// - the unpacked tile is staged SUB-BLOCK-MAJOR ([sjl][16 rows][16 B]): a
//   linear [row][128] layout has a 32-word row stride, which would land all
//   eight gid lanes of a fragment read on one bank (8-way serialization);
//   sub-block-major makes the fragment read's 32 addresses hit 32 banks.

__device__ static uint32_t q6w(uint32_t l, uint32_t h, int hin, int sh) {
    uint32_t lo = hin ? (l >> 4) & 0x0F0F0F0Fu : l & 0x0F0F0F0Fu;
    return __vsub4(lo | (((h >> sh) & 0x03030303u) << 4), 0x20202020u);
}

template<int COLS>
__global__ static void __launch_bounds__(256)
matmul_q6k_mma_kernel(float *out, const unsigned char *wbase, int ts,
                      const int8_t *xq, const float2 *xds, int k, int m) {
    extern __shared__ unsigned char sh[];
    int8_t *sB    = (int8_t *)sh;                                  // [2 bufs][COLS cols][256]
    float2 *sBxds = (float2 *)(sh + 2 * COLS * SB_COL);            // [2 bufs][COLS cols][8]
    int8_t *sA    = (int8_t *)(sh + 2 * COLS * SB_COL + 2 * COLS * SBX * 8); // [warps][2 tiles][8 sjl][16][16]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = blockIdx.x * (32 * (blockDim.x >> 5)) + warp * 32;
    const int gid = lane >> 2, tid = lane & 3;
    const int nbk = k / 256;
    xq  += (size_t)blockIdx.y * COLS * k;             // gridDim.y column-tile (wide chunk); y==0 => identical
    xds += (size_t)blockIdx.y * COLS * (k / 32);
    out += (size_t)blockIdx.y * COLS * m;
    int8_t *sAw = sA + warp * 2 * 2048;
    float acc[2][COLS / 8][4];                                     // [tile][h][slot]
    #pragma unroll
    for (int t = 0; t < 2; t++) for (int h = 0; h < COLS / 8; h++) for (int s = 0; s < 4; s++) acc[t][h][s] = 0.0f;

    // this lane's header row: lanes 0..15 hold tile 0's rows, 16..31 tile 1's
    int rowL = r0 + lane; if (rowL >= m) rowL = m - 1;
    const unsigned char *myrow = wbase + (size_t)rowL * nbk * ts;

    mma_stage_b<COLS>(sB, sBxds, 0, xq, xds, k, 0, threadIdx.x);   // prologue: block 0 in flight
    asm volatile("cp.async.commit_group;\n");

    for (int blk = 0; blk < nbk; blk++) {
        int buf = blk & 1;
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();                                           // staged data visible CTA-wide
        if (blk + 1 < nbk) {                                       // next block overlaps this one's mma
            mma_stage_b<COLS>(sB, sBxds, buf ^ 1, xq, xds, k, blk + 1, threadIdx.x);
            asm volatile("cp.async.commit_group;\n");
        }
        const int8_t *sBb = sB + buf * COLS * SB_COL;
        const float2 *sBxd = sBxds + buf * COLS * SBX;

        const block_q6_Kr *hp = (const block_q6_Kr *)(myrow + (size_t)blk * ts);
        uint32_t sw[4];                                            // my row's 16 int8 scales
        sw[0] = ((const uint32_t *)hp->scales)[0]; sw[1] = ((const uint32_t *)hp->scales)[1];
        sw[2] = ((const uint32_t *)hp->scales)[2]; sw[3] = ((const uint32_t *)hp->scales)[3];
        float dF = d_fp16(hp->d);

        #pragma unroll
        for (int ni = 0; ni < 2; ni++) {                           // 128-element halves
            // stage: lane (ar, hi) unpacks 32 ql bytes + their qh bits for
            // both nibble halves — low nibbles are sub-blocks hi*2/+1, high
            // nibbles hi*2+4/+5 (same bytes, shifted) — into both tiles
            #pragma unroll
            for (int t = 0; t < 2; t++) {
                int ar = lane & 15, hi = lane >> 4;
                int row = r0 + t * 16 + ar; if (row >= m) row = m - 1;
                const block_q6_Kr *bp = (const block_q6_Kr *)(wbase + (size_t)row * nbk * ts) + blk;
                const uint8_t *qlp = bp->ql + ni * 64 + hi * 32;
                const uint8_t *qhp = bp->qh + ni * 32;
                int8_t *sAt = sAw + t * 2048 + ar * 16;
                int shl = hi * 2;
                #pragma unroll
                for (int c = 0; c < 2; c++) {                      // 16-byte halves of the 32
                    uint32_t *plo = (uint32_t *)(sAt + (hi * 2 + c) * 256);
                    uint32_t *phi = (uint32_t *)(sAt + (hi * 2 + 4 + c) * 256);
                    #pragma unroll
                    for (int wi = 0; wi < 4; wi++) {
                        uint32_t l = ld32(qlp + c * 16 + wi * 4), q = ld32(qhp + c * 16 + wi * 4);
                        plo[wi] = q6w(l, q, 0, shl);
                        phi[wi] = q6w(l, q, 1, shl + 4);
                    }
                }
            }
            __syncwarp();
            #pragma unroll
            for (int sjl = 0; sjl < 8; sjl++) {                    // one k16 sub-block per mma
                int sj = ni * 8 + sjl;
                float dscL = dF * (int8_t)(sw[sj >> 2] >> (8 * (sj & 3)));
                float dsc[2][2];                                   // [tile][row-half]
                #pragma unroll
                for (int t = 0; t < 2; t++) {
                    dsc[t][0] = __shfl_sync(0xffffffffu, dscL, t * 16 + gid);
                    dsc[t][1] = __shfl_sync(0xffffffffu, dscL, t * 16 + gid + 8);
                }
                int ga = ni * 4 + (sjl >> 1);                      // activation 32-group
                #pragma unroll
                for (int h = 0; h < COLS / 8; h++) {
                    int colb = h * 8 + gid;
                    uint32_t b0 = *(uint32_t *)(sBb + colb * SB_COL + ni * 128 + sjl * 16 + tid * 4);
                    float2 xd0 = sBxd[(h * 8 + tid * 2) * SBX + ga];
                    float2 xd1 = sBxd[(h * 8 + tid * 2 + 1) * SBX + ga];
                    #pragma unroll
                    for (int t = 0; t < 2; t++) {                  // both tiles ride one B load
                        const int8_t *sAt = sAw + t * 2048 + sjl * 256;
                        uint32_t a0 = *(uint32_t *)(sAt + gid * 16 + tid * 4);
                        uint32_t a1 = *(uint32_t *)(sAt + (gid + 8) * 16 + tid * 4);
                        int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                        asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                            "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                            : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                            : "r"(a0), "r"(a1), "r"(b0));
                        acc[t][h][0] += xd0.x * dsc[t][0] * (float)c0;
                        acc[t][h][1] += xd1.x * dsc[t][0] * (float)c1;
                        acc[t][h][2] += xd0.x * dsc[t][1] * (float)c2;
                        acc[t][h][3] += xd1.x * dsc[t][1] * (float)c3;
                    }
                }
            }
            __syncwarp();                                          // all reads done before re-stage
        }
    }

    #pragma unroll
    for (int t = 0; t < 2; t++)
        for (int h = 0; h < COLS / 8; h++) {
            int col0 = h * 8 + tid * 2;
            int rA = r0 + t * 16 + gid, rB = r0 + t * 16 + gid + 8;
            if (rA < m) { out[(size_t)col0 * m + rA] = acc[t][h][0]; out[(size_t)(col0 + 1) * m + rA] = acc[t][h][1]; }
            if (rB < m) { out[(size_t)col0 * m + rB] = acc[t][h][2]; out[(size_t)(col0 + 1) * m + rB] = acc[t][h][3]; }
        }
}

// Chunk-matmul work by weight type (MACs), for the mma-kernel coverage
// question: which types must the tensor-core path handle to matter?
static double g_cov[40];
static void matmul_coverage_print(void) {
    double tot = 0;
    for (int i = 0; i < 40; i++) tot += g_cov[i];
    if (tot <= 0) return;
    fprintf(stderr, "chunk matmul coverage by type (%% of MACs):");
    for (int i = 0; i < 40; i++)
        if (g_cov[i] > 0) fprintf(stderr, "  type%d %.1f%%", i, 100.0 * g_cov[i] / tot);
    fprintf(stderr, "\n");
}

// ==== TIER 2: faithful line-by-line port of llama.cpp's q4_K MMQ (LG_MMQ_PORT) ====
// Verbatim mma primitives + load_tiles_q4_K + vec_dot_q8_1_q8_1_mma + write_back +
// the block_q8_1_mmq activation, transcribed from ../llama.cpp (Ampere/Turing MMA).
// f64-validated standalone in .scratch/mmq_port.cu (err 3.6e-6 / 0.0026). Activation
// repacked from our g_xq/g_xds. ntx=1; COLS = the chunk width (a multiple of 8).
#define PMMQ_NE_K 32
#define PMMQ_QI8_1 8
#define PMMQ_X_K   (2*PMMQ_NE_K + 2*PMMQ_NE_K/8 + 4)   // 76
#define PMMQ_Y_K   (PMMQ_NE_K + PMMQ_NE_K/PMMQ_QI8_1)  // 36
#define PMMQ_Y     128
#define PMMQ_NWARP 8
struct block_q8_1_mmq { half2 ds4[4]; int8_t qs[4*32]; };

struct pt168 { static constexpr int I=16,J=8,ne=4; int x[4]={0,0,0,0};
    static __device__ __forceinline__ int get_i(int l){return ((l/2)*8)+(threadIdx.x/4);}
    static __device__ __forceinline__ int get_j(int l){return ((threadIdx.x%4)*2)+(l%2);} };
struct pt88  { static constexpr int I=8,J=8,ne=2; int x[2]={0,0};
    static __device__ __forceinline__ int get_i(int l){return threadIdx.x/4;}
    static __device__ __forceinline__ int get_j(int l){return (l*4)+(threadIdx.x%4);} };
__device__ __forceinline__ void pld_ldm(pt168 &t,const int *xs0,int stride){
    int *xi=t.x; const int *xs=xs0+(threadIdx.x%t.I)*stride+(threadIdx.x/t.I)*(t.J/2);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
        :"=r"(xi[0]),"=r"(xi[1]),"=r"(xi[2]),"=r"(xi[3]):"l"(xs)); }
__device__ __forceinline__ void pld_gen(pt88 &t,const int *xs0,int stride){
    #pragma unroll
    for(int l=0;l<t.ne;l++) t.x[l]=xs0[t.get_i(l)*stride+t.get_j(l)]; }
__device__ __forceinline__ void pmma(pt168 &D,const pt168 &A,const pt88 &B){
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        :"+r"(D.x[0]),"+r"(D.x[1]),"+r"(D.x[2]),"+r"(D.x[3])
        :"r"(A.x[0]),"r"(A.x[1]),"r"(A.x[2]),"r"(A.x[3]),"r"(B.x[0]),"r"(B.x[1])); }
__device__ __forceinline__ int punpack_sc(const int *s,int ksc){
    return ((s[(ksc%2)+(ksc!=0)] >> (4*(ksc&(ksc/2)))) & 0x0F0F0F0F)
         | ((s[ksc/2]            >> (2*(ksc%2)))       & 0x30303030); }

__device__ void pload_tiles_q4K(const block_q4_K *x,int *x_tile,int kbx0,int stride){
    int *x_qs=x_tile; half2 *x_dm=(half2*)(x_qs+2*PMMQ_NE_K);
    const int txi=threadIdx.x;
    #pragma unroll
    for(int i0=0;i0<PMMQ_Y;i0+=PMMQ_NWARP){ int i=i0+threadIdx.y;
        const block_q4_K *bxi=x+kbx0+i*stride;
        const int qs0=*((const int*)bxi->qs + txi);
        x_qs[i*PMMQ_X_K + 16*(txi/8)+txi%8+0]=(qs0>>0)&0x0F0F0F0F;
        x_qs[i*PMMQ_X_K + 16*(txi/8)+txi%8+8]=(qs0>>4)&0x0F0F0F0F; }
    #pragma unroll
    for(int i0=0;i0<PMMQ_Y;i0+=PMMQ_NWARP*16){
        int i=(i0+threadIdx.y*16+threadIdx.x/2)%PMMQ_Y;
        const block_q4_K *bxi=x+kbx0+i*stride; const int *sc=(const int*)bxi->scales;
        const int ksc=threadIdx.x%2;
        const int sc32=punpack_sc(sc,ksc+0), m32=punpack_sc(sc,ksc+2);
        const uint8_t *s8=(const uint8_t*)&sc32; const uint8_t *m8=(const uint8_t*)&m32;
        const half2 dm=*(const half2*)&bxi->d * make_half2(1.0f,-1.0f);
        #pragma unroll
        for(int l=0;l<4;l++) x_dm[i*PMMQ_X_K + sizeof(int)*ksc + l]=dm*make_half2(s8[l],m8[l]); }
}
// ntx = x minitiles/warp: wide tiles (COLS>=48) use granularity 16 -> 2 minitiles,
// amortizing each activation B-load across 2 weight A-tiles (matches mmq_get_granularity_device).
#define PMMQ_NTX(COLS) (((COLS)>=48 ? 16 : 8) * 2 / 16)
template<int COLS> __device__ void pvecdot(const int *x,const int *y,float *sum,int k00){
    constexpr int ntx=PMMQ_NTX(COLS);
    const int *x_qs=x; const half2 *x_dm=(const half2*)x_qs+2*PMMQ_NE_K;
    y += (threadIdx.y % ntx)*(8*PMMQ_Y_K);                 // tile_C::J=8
    const int *y_qs=(const int*)y+4; const half2 *y_dm=(const half2*)y;
    pt168 A[ntx][4]; float2 dmA[ntx][2][4]; const int i0=(threadIdx.y/ntx)*(ntx*16);
    #pragma unroll
    for(int n=0;n<ntx;n++){
        #pragma unroll
        for(int k01=0;k01<PMMQ_NE_K;k01+=PMMQ_QI8_1) pld_ldm(A[n][k01/PMMQ_QI8_1], x_qs+(i0+n*16)*PMMQ_X_K+(k00+k01), PMMQ_X_K);
        #pragma unroll
        for(int l=0;l<2;l++){ int i=i0+n*16+pt168::get_i(2*l);
            #pragma unroll
            for(int k01=0;k01<PMMQ_NE_K;k01+=PMMQ_QI8_1) dmA[n][l][k01/PMMQ_QI8_1]=__half22float2(x_dm[i*PMMQ_X_K+(k00+k01)/PMMQ_QI8_1]); }
    }
    #pragma unroll
    for(int j0=0;j0<COLS;j0+=ntx*8){
        #pragma unroll
        for(int k01=0;k01<PMMQ_NE_K;k01+=PMMQ_QI8_1){
            pt88 B; float2 dsB[2]; pld_gen(B, y_qs+j0*PMMQ_Y_K+k01, PMMQ_Y_K);
            #pragma unroll
            for(int l=0;l<2;l++){ int j=j0+pt168::get_j(l); dsB[l]=__half22float2(y_dm[j*PMMQ_Y_K+k01/PMMQ_QI8_1]); }
            #pragma unroll
            for(int n=0;n<ntx;n++){
                pt168 C; pmma(C,A[n][k01/PMMQ_QI8_1],B);
                #pragma unroll
                for(int l=0;l<4;l++){ sum[(j0/8+n)*4+l]+=dmA[n][l/2][k01/PMMQ_QI8_1].x*dsB[l%2].x*C.x[l];
                                      sum[(j0/8+n)*4+l]+=dmA[n][l/2][k01/PMMQ_QI8_1].y*dsB[l%2].y; } }
        }
    }
}
template<int COLS> __global__ void __launch_bounds__(PMMQ_NWARP*32,1)
mmq_port_kernel(const block_q4_K *x,const int *y,float *dst,int nb,int ncols,int m){
    extern __shared__ int psh[];
    int *tile_y=psh; int *tile_x=tile_y + COLS*PMMQ_Y_K;
    const int rowTile=blockIdx.x*PMMQ_Y, colTile=blockIdx.y*COLS;
    const int tix=threadIdx.y*32+threadIdx.x;
    float sum[COLS*PMMQ_Y/(PMMQ_NWARP*32)]={0};
    const int sz=sizeof(block_q8_1_mmq)/sizeof(int);
    for(int kb0=0;kb0<nb;kb0++){
        pload_tiles_q4K(x+(size_t)rowTile*nb, tile_x, kb0, nb);
        { const int *by0=y + (size_t)(colTile + (kb0*2)*ncols)*sz;
          for(int l=tix;l<COLS*PMMQ_Y_K;l+=PMMQ_NWARP*32) tile_y[l]=by0[l]; }
        __syncthreads(); pvecdot<COLS>(tile_x,tile_y,sum,0); __syncthreads();
        { const int *by0=y + (size_t)(colTile + (kb0*2+1)*ncols)*sz;
          for(int l=tix;l<COLS*PMMQ_Y_K;l+=PMMQ_NWARP*32) tile_y[l]=by0[l]; }
        __syncthreads(); pvecdot<COLS>(tile_x,tile_y,sum,PMMQ_NE_K); __syncthreads();
    }
    constexpr int ntx=PMMQ_NTX(COLS);
    const int i0=(threadIdx.y/ntx)*(ntx*16);
    #pragma unroll
    for(int j0=0;j0<COLS;j0+=ntx*8)
        #pragma unroll
        for(int n=0;n<ntx;n++)
            #pragma unroll
            for(int l=0;l<4;l++){ int j=colTile+j0+(threadIdx.y%ntx)*8+pt168::get_j(l), i=rowTile+i0+n*16+pt168::get_i(l);
                if(i<m) dst[(size_t)j*m + i]=sum[(j0/8+n)*4+l]; }
}
// repack our activation (g_xq int8 + g_xds (d_y, sum_q)) -> block_q8_1_mmq y[(k/128)*cols+col]
__global__ static void pmmq_repack(block_q8_1_mmq *y,const int8_t *xq,const float2 *xds,int k,int cols){
    int c=blockIdx.x;
    for(int g=threadIdx.x; g<k/32; g+=blockDim.x){
        int kb=g/4, s=g%4; float2 ds=xds[(size_t)c*(k/32)+g];
        block_q8_1_mmq *b=&y[(size_t)kb*cols + c];
        b->ds4[s]=make_half2(ds.x, ds.x*ds.y);            // (d_y, sum_float = d_y*sum_q)
        const int8_t *src=xq + (size_t)c*k + g*32;
        #pragma unroll
        for(int t=0;t<32;t++) b->qs[s*32+t]=src[t]; }
}
static block_q8_1_mmq *g_ymmq=NULL; static size_t g_ymmq_cap=0;
template<int COLS>
static void launch_q4k_port(float *d_out,const unsigned char *w,const int8_t *xq,const float2 *xds,int k,int m){
    size_t need=(size_t)(k/128)*COLS*sizeof(block_q8_1_mmq);
    if(need>g_ymmq_cap){ cudaFree(g_ymmq); if(cudaMalloc(&g_ymmq,need)!=cudaSuccess){g_ymmq_cap=0;return;} g_ymmq_cap=need; }
    pmmq_repack<<<COLS, 256, 0, g_launch>>>(g_ymmq, xq, xds, k, COLS);
    size_t shm=(size_t)(COLS*PMMQ_Y_K + PMMQ_Y*PMMQ_X_K)*4;
    static int carve=0; if(!carve){ if(shm>48*1024) cudaFuncSetAttribute(mmq_port_kernel<COLS>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shm); carve=1; }
    dim3 grid((m+PMMQ_Y-1)/PMMQ_Y, 1), blk(32,PMMQ_NWARP);
    mmq_port_kernel<COLS><<<grid, blk, shm, g_launch>>>((const block_q4_K*)w, (const int*)g_ymmq, d_out, k/256, COLS, m);
}

// ---- TIER 2 experiment: faithful "load_tiles" tiled q4_K MMQ (LG_MMQ_TILED) ----
// The structural alternative to matmul_q4k_mmq_kernel: unpack the WHOLE 256-K
// weight superblock to shared ONCE per block (a row's 256 nibbles -> sWq[64 ints]
// + the folded (d*sc, -dmin*m) scales -> sWdm[8]), reuse it across the COLS batch
// tile, plain __syncthreads (NO cp.async), 8 warps tiling 128 rows at 1 CTA/128-row
// tile — llama.cpp's load_tiles shape, the piece our fat tile never adopted.
// The occupancy-trap micro dismissed this at 0.35x, but that micro is L2-cached and
// LIES (it dismissed the fat tile too); this measures it on the real Orin prefill.
// Math is byte-equivalent to the fat tile (same nib4 / d_dm / d_gsm32r / combine);
// only the staging differs. Ported from .scratch/mmq_gemm_test.cu (f64-validated).
template<int COLS>
__global__ static void __launch_bounds__(256, 1)
matmul_q4k_tiled_kernel(float *out, const unsigned char *wbase, int ts,
                        const int8_t *xq, const float2 *xds, int k, int m) {
    extern __shared__ unsigned char sh[];
    const int TILE_M = 128, SW = 68, SX = 68;             // PADDED strides (68, off the 32-bank
    int    *sWq  = (int *)sh;                             // multiple) -> conflict-free fragment
    float2 *sWdm = (float2 *)(sWq + TILE_M * SW);         // reads; 64 of the 68 ints/row are real
    int    *sXq  = (int *)(sWdm + TILE_M * 8);            // (256 int8 = 64 ints). sWdm/sXds dense.
    float2 *sXds = (float2 *)(sXq + COLS * SX);           // [COLS][8] (d_y, sum_y)
    const int rowTile = blockIdx.x * TILE_M;
    const int warp = threadIdx.y, lane = threadIdx.x, tix = warp * 32 + lane;
    const int gid = lane >> 2, tid = lane & 3, nb = k / 256;
    const size_t rbytes = (size_t)nb * ts;                // bytes per weight row (in-place q4_K: ts=144)

    float acc[COLS / 8][4];
    #pragma unroll
    for (int j = 0; j < COLS / 8; j++) for (int s = 0; s < 4; s++) acc[j][s] = 0.0f;

    for (int blk = 0; blk < nb; blk++) {
        for (int t = tix; t < TILE_M * 64; t += 256) {    // 64 real ints/row (row-clamped to m)
            int r = t >> 6, ii = t & 63, sj = ii >> 3, i = ii & 7, g = sj >> 1;
            int gr = rowTile + r; if (gr >= m) gr = m - 1;
            const block_q4_K *bp = (const block_q4_K *)(wbase + (size_t)gr * rbytes) + blk;
            uint32_t q4 = *(const uint32_t *)(bp->qs + g * 32 + i * 4);
            sWq[r * SW + ii] = nib4(q4, sj & 1);          // padded write stride
        }
        for (int t = tix; t < TILE_M * 8; t += 256) {
            int r = t >> 3, sj = t & 7;
            int gr = rowTile + r; if (gr >= m) gr = m - 1;
            const block_q4_K *bp = (const block_q4_K *)(wbase + (size_t)gr * rbytes) + blk;
            const uint32_t *s = (const uint32_t *)bp->scales;
            uint8_t sc, mm; d_gsm32r(sj, s[0], s[1], s[2], &sc, &mm);
            float dD, dM; d_dm(&bp->d, &dD, &dM);
            sWdm[t] = make_float2(dD * sc, dM * mm);
        }
        for (int t = tix; t < COLS * 64; t += 256) {      // 64 real ints/col q8_1 activation
            int c = t >> 6, ii = t & 63;
            sXq[c * SX + ii] = *(const int *)(xq + (size_t)c * k + blk * 256 + ii * 4);
        }
        for (int t = tix; t < COLS * 8; t += 256) {
            int c = t >> 3, sj = t & 7;
            sXds[t] = xds[(size_t)c * (k / 32) + blk * 8 + sj];
        }
        __syncthreads();
        #pragma unroll
        for (int sj = 0; sj < 8; sj++) {
            const int rA = warp * 16 + gid, rB = warp * 16 + gid + 8;
            uint32_t a0 = sWq[rA*SW + sj*8 + tid],     a1 = sWq[rB*SW + sj*8 + tid];
            uint32_t a2 = sWq[rA*SW + sj*8 + tid + 4], a3 = sWq[rB*SW + sj*8 + tid + 4];
            float2 dmA = sWdm[rA*8 + sj], dmB = sWdm[rB*8 + sj];
            #pragma unroll
            for (int jt = 0; jt < COLS / 8; jt++) {
                int cB = jt*8 + gid;
                uint32_t b0 = sXq[cB*SX + sj*8 + tid], b1 = sXq[cB*SX + sj*8 + tid + 4];
                int col0 = jt*8 + tid*2;
                float2 xd0 = sXds[col0*8 + sj], xd1 = sXds[(col0 + 1)*8 + sj];
                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                    : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                acc[jt][0] += xd0.x * (dmA.x * c0 - dmA.y * xd0.y);
                acc[jt][1] += xd1.x * (dmA.x * c1 - dmA.y * xd1.y);
                acc[jt][2] += xd0.x * (dmB.x * c2 - dmB.y * xd0.y);
                acc[jt][3] += xd1.x * (dmB.x * c3 - dmB.y * xd1.y);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int jt = 0; jt < COLS / 8; jt++) {
        int col0 = jt*8 + tid*2, rA = rowTile + warp*16 + gid, rB = rowTile + warp*16 + gid + 8;
        if (rA < m) { out[(size_t)col0*m + rA] = acc[jt][0]; out[(size_t)(col0 + 1)*m + rA] = acc[jt][1]; }
        if (rB < m) { out[(size_t)col0*m + rB] = acc[jt][2]; out[(size_t)(col0 + 1)*m + rB] = acc[jt][3]; }
    }
}

template<int COLS>
static void launch_q4k_tiled(float *d_out, const unsigned char *w, int ts, const int8_t *xq, const float2 *xds, int k, int m) {
    const int TILE_M = 128, SW = 68, SX = 68;             // padded strides (must match the kernel)
    size_t shm = (size_t)TILE_M*SW*4 + (size_t)TILE_M*8*sizeof(float2) + (size_t)COLS*SX*4 + (size_t)COLS*8*sizeof(float2);
    static int carve = 0;
    if (!carve) { if (shm > 48*1024) cudaFuncSetAttribute(matmul_q4k_tiled_kernel<COLS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm); carve = 1; }
    int blocks = (m + TILE_M - 1) / TILE_M;
    matmul_q4k_tiled_kernel<COLS><<<dim3(blocks), dim3(32, TILE_M/16), shm, g_launch>>>(d_out, w, ts, xq, xds, k, m);
}

// Launch the fat q4_K kernel for a compile-time COLS (= the adaptive chunk
// width g_pf_cols). Each COLS instantiation opts into its own shared carveout
// once (COLS>=64 needs >48KB). 16 rows/warp; wpc shrinks so the grid covers SMs.
template<int COLS>
static void launch_q4k_mmq(float *d_out, const unsigned char *w, int ts, const int8_t *xq, const float2 *xds, int k, int m, int sms, int ncol = 1) {
    int wpc = 8;
    while (wpc > 1 && (long)((m + 16 * wpc - 1) / (16 * wpc)) * ncol < 2 * sms) wpc >>= 1;  // ncol col-tiles also fill SMs
    size_t shm = 2 * COLS * SB_COL + 2 * COLS * SBX * sizeof(float2) + (size_t)wpc * 2 * 16 * SA_ROW;
    static int carve = 0;
    if (!carve) {
        size_t maxshm = 2 * COLS * SB_COL + 2 * COLS * SBX * sizeof(float2) + (size_t)8 * 2 * 16 * SA_ROW;
        if (maxshm > 48 * 1024) cudaFuncSetAttribute(matmul_q4k_mmq_kernel<COLS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxshm);
        carve = 1;
    }
    int blocks = (m + 16 * wpc - 1) / (16 * wpc);
    matmul_q4k_mmq_kernel<COLS><<<dim3(blocks, ncol), 32 * wpc, shm, g_launch>>>(d_out, w, ts, xq, xds, k, m);
}

// 2-row-tile q4_K launcher (LG_Q4K_2TILE): 32 rows/warp, COLS=64 cols, sA holds
// two tiles. Halves per-mma B/scale shared traffic (the L1-throughput cap on the
// big FFN). ncol column-tiles over gridDim.y like the 1-tile path.
template<int COLS>
static void launch_q4k_mmq2(float *d_out, const unsigned char *w, int ts, const int8_t *xq, const float2 *xds, int k, int m, int sms, int ncol) {
    int wpc = 8;
    while (wpc > 1 && (long)((m + 32 * wpc - 1) / (32 * wpc)) * ncol < 2 * sms) wpc >>= 1;  // 32 rows/warp
    size_t shm = 2 * COLS * SB_COL + 2 * COLS * SBX * sizeof(float2) + (size_t)wpc * 2 * 2 * 16 * SA_ROW;
    static int carve = 0;
    if (!carve) {
        size_t maxshm = 2 * COLS * SB_COL + 2 * COLS * SBX * sizeof(float2) + (size_t)8 * 2 * 2 * 16 * SA_ROW;
        if (maxshm > 48 * 1024) cudaFuncSetAttribute(matmul_q4k_mmq2_kernel<COLS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxshm);
        carve = 1;
    }
    int blocks = (m + 32 * wpc - 1) / (32 * wpc);
    matmul_q4k_mmq2_kernel<COLS><<<dim3(blocks, ncol), 32 * wpc, shm, g_launch>>>(d_out, w, ts, xq, xds, k, m);
}

// Stream-K stage 1 launcher (LG_Q4K_SK): grid = nsm persistent CTAs (wpc fixed 8 =
// 256-row tiles, 1 CTA/SM), each grinds a strided slice of the ntx*nty work-list.
template<int COLS>
static void launch_q4k_sk(float *d_out, const unsigned char *w, int ts, const int8_t *xq, const float2 *xds, int k, int m, int sms, int ntx) {
    const int wpc = 8;
    size_t shm = 2 * COLS * SB_COL + 2 * COLS * SBX * sizeof(float2) + (size_t)wpc * 2 * 2 * 16 * SA_ROW;
    static int carve = 0;
    if (!carve) {
        if (shm > 48 * 1024) cudaFuncSetAttribute(matmul_q4k_sk_kernel<COLS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
        carve = 1;
    }
    long nty = (m + 32 * wpc - 1) / (32 * wpc);
    long ntiles = nty * ntx;
    int grid = (int)(ntiles < sms ? ntiles : sms);            // persistent: <= nsm CTAs
    matmul_q4k_sk_kernel<COLS><<<grid, 32 * wpc, shm, g_launch>>>(d_out, w, ts, xq, xds, k, m, ntx);
}

// q6_K twin launcher for a compile-time COLS. The q6_K mma kernel stays 2-tile
// (32 rows/warp), whose acc[2][COLS/8][4] fits up to COLS=64 (~121 regs); the big
// 2*2048/warp A tile needs the carveout past COLS=32. Widening the q6_K sub-tile
// 32->64 halves its weight passes (the wide-tile win, applied to the 9%-of-prefill
// q6_K share without a 1-tile rewrite).
template<int COLS>
static void launch_q6k_mmq(float *d_out, const unsigned char *w, int ts, const int8_t *xq, const float2 *xds, int k, int m, int sms, int ncol = 1) {
    int wpc = 8;
    while (wpc > 1 && (long)((m + 32 * wpc - 1) / (32 * wpc)) * ncol < 2 * sms) wpc >>= 1;   // cover SMs (32 rows/warp)
    size_t shm = 2 * COLS * SB_COL + 2 * COLS * SBX * sizeof(float2) + (size_t)wpc * 2 * 2048;
    static int carve = 0;
    if (!carve) {
        size_t maxshm = 2 * COLS * SB_COL + 2 * COLS * SBX * sizeof(float2) + (size_t)8 * 2 * 2048;
        if (maxshm > 48 * 1024) cudaFuncSetAttribute(matmul_q6k_mma_kernel<COLS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxshm);
        carve = 1;
    }
    int blocks = (m + 32 * wpc - 1) / (32 * wpc);
    matmul_q6k_mma_kernel<COLS><<<dim3(blocks, ncol), 32 * wpc, shm, g_launch>>>(d_out, w, ts, xq, xds, k, m);
}

static void matmul_q_n(float *d_out, const struct gguf_tensor *t, const float *d_x, int k, int m) {
    rweight_init_all();
    int blck = ggml_blck_size(t->type), ts;
    const unsigned char *w = rweight(t, &ts);
    if (t->type < 40) g_cov[t->type] += (double)k * m;
    static int no_mma = -1;
    if (no_mma < 0) no_mma = getenv("LG_NO_MMA") != NULL;
    static int tiled = -1;                                   // TIER 2: faithful load_tiles q4_K path
    if (tiled < 0) tiled = getenv("LG_MMQ_TILED") != NULL;
    static int port = -1;                                    // TIER 2: full line-by-line llama.cpp MMQ port
    if (port < 0) port = getenv("LG_MMQ_PORT") != NULL;
    static int t2 = -1;                                      // 2-row-tile q4_K (B/scale reuse across 2 tiles)
    if (t2 < 0) t2 = getenv("LG_Q4K_2TILE") != NULL;
    static int sk = -1;                                      // stream-K stage 1: persistent nsm CTAs (2-tile body)
    if (sk < 0) sk = getenv("LG_Q4K_SK") != NULL;
    static int sms = 0;
    if (!sms) { cudaDeviceProp p; cudaGetDeviceProperties(&p, 0); sms = p.multiProcessorCount; }

    // q4_K: fat-tile MMQ — one wide launch over the chunk's columns, in-place (no
    // REPACK twin). One 16-row tile per warp -> acc[COLS/8][4]; eight warps tile
    // 128 rows at ~1 CTA/SM, the carveout + launch_bounds(256,1) regime the Orin
    // measurement showed beats llama.cpp's at half our occupancy (ILP over the
    // wide accumulator hides the LDS latency). COLS is the adaptive chunk width
    // g_pf_cols (32/64/96/128): full chunks 128 (4x the old 32-wide reuse), the
    // short tail of a turn smaller so it doesn't pay a 128-wide chunk.
    if (t->type == GGML_TYPE_Q4_K && PREFILL_B % 32 == 0 && !no_mma) {
        // Wide chunk: tile the COLS=128 columns across gridDim.y in ONE launch so a
        // row-tile's q4_K weights stay L2-hot across its column-tiles (llama.cpp's
        // 512-ubatch weight reuse), instead of re-streaming them per 128-col launch.
        // At g_pf_cols==128 this is ncol=1, grid.y==1 == the legacy single launch.
        if (!port && !tiled) {
            // stream-K (LG_Q4K_SK): persistent nsm CTAs grind the 2-tile work-list.
            if (sk && g_pf_cols >= 64 && g_pf_cols % 64 == 0) {
                launch_q4k_sk<64>(d_out, w, ts, g_xq, g_xds, k, m, sms, g_pf_cols / 64);
                return;
            }
            // 2-row-tile (LG_Q4K_2TILE): COLS=64, each B+scale load feeds 2 mmas ->
            // half the per-mma shared traffic that caps the 1-tile big FFN at 44% SM.
            if (t2 && g_pf_cols >= 64 && g_pf_cols % 64 == 0) {
                launch_q4k_mmq2<64>(d_out, w, ts, g_xq, g_xds, k, m, sms, g_pf_cols / 64);
                return;
            }
            if (g_pf_cols >= 128 && g_pf_cols % 128 == 0) {
                launch_q4k_mmq<128>(d_out, w, ts, g_xq, g_xds, k, m, sms, g_pf_cols / 128);
                return;
            }
        }
        // The fat kernel templates COLS<=128 (acc[COLS/8][4] regs). A whole-image
        // chunk can be wider (bidirectional spans, stage 2), so tile the columns in
        // <=128 passes over offset slices of g_xq/d_out. g_pf_cols is a multiple of
        // 32, so each tile is too -> the {32,64,96,128} templates cover it exactly.
        // At g_pf_cols<=128 this is a single pass = the old single launch.
        for (int c0 = 0; c0 < g_pf_cols; c0 += 128) {
            int ct = g_pf_cols - c0; if (ct > 128) ct = 128;
            const int8_t *xqc = g_xq + (size_t)c0 * k; const float2 *xdc = g_xds + (size_t)c0 * (k / 32);
            float *outc = d_out + (size_t)c0 * m;
            if (port) switch (ct) {                          // full llama.cpp MMQ port (A/B)
                case 32:  launch_q4k_port<32 >(outc, w, xqc, xdc, k, m); break;
                case 64:  launch_q4k_port<64 >(outc, w, xqc, xdc, k, m); break;
                case 96:  launch_q4k_port<96 >(outc, w, xqc, xdc, k, m); break;
                default:  launch_q4k_port<128>(outc, w, xqc, xdc, k, m); break;
            } else if (tiled) switch (ct) {                  // faithful load_tiles path (A/B)
                case 32:  launch_q4k_tiled<32 >(outc, w, ts, xqc, xdc, k, m); break;
                case 64:  launch_q4k_tiled<64 >(outc, w, ts, xqc, xdc, k, m); break;
                case 96:  launch_q4k_tiled<96 >(outc, w, ts, xqc, xdc, k, m); break;
                default:  launch_q4k_tiled<128>(outc, w, ts, xqc, xdc, k, m); break;
            } else switch (ct) {
                case 32:  launch_q4k_mmq<32 >(outc, w, ts, xqc, xdc, k, m, sms); break;
                case 64:  launch_q4k_mmq<64 >(outc, w, ts, xqc, xdc, k, m, sms); break;
                case 96:  launch_q4k_mmq<96 >(outc, w, ts, xqc, xdc, k, m, sms); break;
                default:  launch_q4k_mmq<128>(outc, w, ts, xqc, xdc, k, m, sms); break;
            }
        }
        return;
    }
    // q6_K (native) and the f32/bf16 PLE q6_K twin share the 2-tile mma kernel,
    // whose acc[2][COLS/8][4] spills past COLS=32 — so loop the chunk in 32-col
    // sub-tiles (offset out/xq/xds per slice). At PREFILL_B=32 that's a single
    // iteration = the old single launch; same float order either way.
    const unsigned char *q6src = NULL; int q6ts = 0;
    if (!no_mma) {
        if (t->type == GGML_TYPE_Q6_K) { q6src = w; q6ts = ts; }
        else { const unsigned char *q6 = rweight_q6(t); if (q6) { q6src = q6; q6ts = (int)sizeof(block_q6_Kr); } }
    }
    if (q6src && PREFILL_B % 32 == 0) {
        // Wide chunk: one launch, COLS=64 tiles across gridDim.y -> the row-tile's
        // q6_K weights stay L2-hot across all its column-tiles (this is q6_K's bigger
        // win — its 64-col tile re-streams weights twice as often as q4_K's 128).
        if (g_pf_cols > 128 && g_pf_cols % 64 == 0) {
            launch_q6k_mmq<64>(d_out, q6src, q6ts, g_xq, g_xds, k, m, sms, g_pf_cols / 64);
            return;
        }
        // 64-wide sub-tiles where the chunk allows (halves weight passes vs 32),
        // a 32-wide tail for the leftover 32 cols (g_pf_cols in {32,64,96,128}).
        for (int c0 = 0; c0 < g_pf_cols; ) {
            const int8_t *xqc = g_xq + (size_t)c0 * k; const float2 *xdc = g_xds + (size_t)c0 * (k / 32);
            float *outc = d_out + (size_t)c0 * m;
            if (g_pf_cols - c0 >= 64) { launch_q6k_mmq<64>(outc, q6src, q6ts, xqc, xdc, k, m, sms); c0 += 64; }
            else                     { launch_q6k_mmq<32>(outc, q6src, q6ts, xqc, xdc, k, m, sms); c0 += 32; }
        }
        return;
    }
    // dp4a fallback (q3/q5/q8 and the residual f32/bf16): float s[NB] spills past
    // ~32, so loop 32-col sub-tiles here too.
    int rows_per_block = 256 / 32;
    int blocks = (m + rows_per_block - 1) / rows_per_block;
    for (int c0 = 0; c0 < g_pf_cols; c0 += 32)                   // chunk width (adaptive), 32-col sub-tiles
        matmul_i8r_n_kernel<32><<<blocks, 256, 0, g_launch>>>(
            d_out + (size_t)c0 * m, w, (int)t->type, ts, blck, d_x + (size_t)c0 * k,
            g_xq + (size_t)c0 * k, g_xds + (size_t)c0 * (k / 32), k, m);
}
// The MTP verify matmul: LG_MTP_N query columns in one launch. Byte-identical to
// decode per query — the integer sub-block dots are order-independent for any NB,
// and the NB==LG_MTP_N float-aux guards above keep the float matmuls in decode order.
static void matmul_q_spec(float *d_out, const struct gguf_tensor *t, const float *d_x, int k, int m) {
    rweight_init_all();
    int blck = ggml_blck_size(t->type), ts;
    const unsigned char *w = rweight(t, &ts);
    int blocks = (m + 7) / 8;
    matmul_i8r_n_kernel<LG_MTP_N><<<blocks, 256, 0, g_launch>>>(d_out, w, (int)t->type, ts, blck, d_x, g_xq, g_xds, k, m);
}
