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

// q4_K tile-coalesced repack for the mma prefill kernel (the "MMQ" rewrite).
// In the mapped blob, the 32 rows a warp covers sit one 144-B block apart, so
// the warp's A-staging load scatters across DRAM (~49% wasted sectors on the
// Orin's uncached zero-copy reads). Here RPACK=16 rows share one tile-block and
// their quant bytes are interleaved so consecutive lanes read contiguous bytes
// (one 512-B coalesced group per warp). Scales are PRE-DECODED from the 12-byte
// 6-bit twist to flat uint8 (sc[8] then m[8] per row, no -32 bias); (d,dmin)
// stay fp16. Built as a SECOND device copy (~q4_K size) only when it fits VRAM
// (E4B on a 16 GB Orin; 12B falls back to the in-place mma path). Validated
// byte-lossless + byte-identical to the in-place kernel in .scratch/mmq_*_test.
#define RPACK      16
#define Q4KR_QUANT (RPACK * 128)                          // 2048: [j 0..7][row 0..15][16 B]
#define Q4KR_SCALE (RPACK * 16)                            // 256:  per row, sc[8] then m[8]
#define Q4KR_DM    (RPACK * 4)                             // 64:   per row, (d,dmin) fp16
#define Q4KR_TBLK  (Q4KR_QUANT + Q4KR_SCALE + Q4KR_DM)     // 2368
static_assert(Q4KR_TBLK % 16 == 0, "q4_Kr tile-block 16-aligned");

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

// q4_K 6-bit twisted-scale decode on the host (the device twin is d_gsm32);
// pre-decoded into block_q4_Kr so the mma kernel reads flat uint8 scales.
static void d_gsm32_host(int j, const uint32_t *s, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = (s[0] >> (8 * j)) & 63; *m = (s[1] >> (8 * j)) & 63; }
    else {
        int b = 8 * (j - 4);
        *d = ((s[2] >> b) & 0x0F) | (((s[0] >> (b + 6)) & 3) << 4);
        *m = (((s[2] >> b) >> 4) & 0x0F) | (((s[1] >> (b + 6)) & 3) << 4);
    }
}
// Repack m rows x nbk q4_K blocks into ntile tile-blocks (block_q4_Kr layout).
// ntile is EVEN (= roundup(m,32)/16) so the kernel's two-tile-per-warp reads of
// tile-block tbrow+1 never run past the end; padding rows clamp to row m-1,
// matching the in-place kernel's rowL clamp, so every slot a warp can touch is
// a real weight. Byte-lossless (gate #1) and value-exact in the kernel (gate #2).
static void repack_q4_K_mmq(uint8_t *dst, const block_q4_K *src, int m, int nbk, int ntile) {
    memset(dst, 0, (size_t)ntile * nbk * Q4KR_TBLK);
    for (int tt = 0; tt < ntile; tt++)
      for (int blk = 0; blk < nbk; blk++) {
        uint8_t *tb = dst + ((size_t)tt * nbk + blk) * Q4KR_TBLK;
        uint8_t *qd = tb, *sd = tb + Q4KR_QUANT;
        uint16_t *dd = (uint16_t *)(tb + Q4KR_QUANT + Q4KR_SCALE);
        for (int r = 0; r < RPACK; r++) {
            int row = tt * RPACK + r; if (row >= m) row = m - 1;          // tail clamp
            const block_q4_K *b = &src[(size_t)row * nbk + blk];
            for (int j = 0; j < 8; j++) memcpy(qd + (j * RPACK + r) * 16, b->qs + j * 16, 16);
            uint32_t s[3]; memcpy(s, b->scales, 12);
            for (int sj = 0; sj < 8; sj++) { uint8_t sc, mm; d_gsm32_host(sj, s, &sc, &mm); sd[r * 16 + sj] = sc; sd[r * 16 + 8 + sj] = mm; }
            uint16_t db, mb; memcpy(&db, &b->d, 2); memcpy(&mb, &b->dmin, 2);
            dd[r * 2 + 0] = db; dd[r * 2 + 1] = mb;
        }
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

// Tile-coalesced q4_K twins for the mma prefill kernel (block_q4_Kr). A SECOND
// device copy of every q4_K weight, indexed like g_rw. NULL entry => that
// tensor uses the in-place mma path (REPACK=false, byte-identical). The whole
// table stays NULL when the copy doesn't fit free VRAM (12B on a 16 GB Orin).
static unsigned char **g_rw_mmq = NULL;

static const unsigned char *rweight_mmq(const struct gguf_tensor *t) {
    if (!g_rw_mmq) return NULL;
    return g_rw_mmq[(size_t)(t - g_ctx->tensors)];
}

// Build the q4_K twins up front (like rweight_init_all, so no cudaMalloc lands
// mid graph-capture). All-or-none on a free-VRAM gate with a 1 GB reserve for
// context/scratch growth: E4B's ~4 GB second copy fits a 16 GB Orin, 12B's
// ~5.5 GB does not (it keeps the in-place mma path). Discrete GPUs (lots of
// VRAM) build it for every model. Logs the decision for the device run.
static void mmq_build_all(void) {
    if (getenv("LG_NO_MMQ")) {                            // force the in-place mma path (REPACK=false)
        fprintf(stderr, "MMQ q4_K repack: disabled by LG_NO_MMQ (in-place mma)\n");
        return;
    }
    size_t total = 0; int ntensor = 0;
    for (uint64_t i = 0; i < g_ctx->header.num_tensors; i++) {
        const struct gguf_tensor *t = &g_ctx->tensors[i];
        if (t->type != GGML_TYPE_Q4_K || t->n_dims != 2) continue;
        int k = (int)t->dims[0], m = (int)t->dims[1];
        if (k % QK_K) continue;
        int nbk = k / QK_K, ntile = ((m + 31) / 32) * 2;
        total += (size_t)ntile * nbk * Q4KR_TBLK; ntensor++;
    }
    if (!ntensor) return;
    size_t freeb = 0, totb = 0;
    if (cudaMemGetInfo(&freeb, &totb) != cudaSuccess) { cudaGetLastError(); return; }
    const size_t reserve = (size_t)1 << 30;
    int fits = total + reserve <= freeb;
    fprintf(stderr, "MMQ q4_K repack: %d tensors, %.2f GB; free %.2f GB, reserve 1.00 GB -> %s\n",
            ntensor, total / 1e9, freeb / 1e9, fits ? "ENABLED" : "disabled (in-place mma)");
    if (!fits) return;
    g_rw_mmq = (unsigned char **)calloc(g_ctx->header.num_tensors, sizeof *g_rw_mmq);
    if (!g_rw_mmq) return;
    for (uint64_t i = 0; i < g_ctx->header.num_tensors; i++) {
        const struct gguf_tensor *t = &g_ctx->tensors[i];
        if (t->type != GGML_TYPE_Q4_K || t->n_dims != 2) continue;
        int k = (int)t->dims[0], m = (int)t->dims[1];
        if (k % QK_K) continue;
        int nbk = k / QK_K, ntile = ((m + 31) / 32) * 2;
        size_t bytes = (size_t)ntile * nbk * Q4KR_TBLK;
        uint8_t *host = (uint8_t *)malloc(bytes);
        if (!host) { fprintf(stderr, "MMQ: host OOM on %s; in-place for it\n", t->name); continue; }
        repack_q4_K_mmq(host, (const block_q4_K *)t->data, m, nbk, ntile);
        unsigned char *dev;
        if (cudaMalloc(&dev, bytes) != cudaSuccess) { cudaGetLastError(); free(host); fprintf(stderr, "MMQ: device OOM on %s; in-place for it\n", t->name); continue; }
        cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice);
        free(host);
        g_rw_mmq[(size_t)i] = dev;
    }
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
    mmq_build_all();
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
        if (NB == 2 || (k & 3)) {                        // verify pair: decode's order, exactly
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
        if (NB == 2 || (k & 7)) {                        // one uint32 = 2 elements
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

// Plain launch bounds on purpose: capping to 64 regs for a 4th CTA (and the
// max shared carveout that would seat it) BOTH regressed the Orin — tighter
// scheduling and a smaller L1 cost more than the occupancy bought. 80 regs,
// 3 CTAs, 43% occupancy is this shape's measured optimum; round 4's notes
// have the numbers.
// REPACK=true reads the tile-coalesced block_q4_Kr twin (wbase = g_rw_mmq[t]);
// REPACK=false reads the mapped q4_K blob in place (== the committed kernel,
// byte-identical — gate #2). Only the two global reads (A-staging quant fetch +
// per-row scale source) differ; the staging layout, mma, shuffle, and float
// order are shared, so the two instantiations produce identical output.
template<int COLS, bool REPACK>
__global__ static void __launch_bounds__(256)
matmul_q4k_mma_kernel(float *out, const unsigned char *wbase, int ts,
                      const int8_t *xq, const float2 *xds, int k, int m, float *pacc) {
    extern __shared__ unsigned char sh[];
    int8_t *sB    = (int8_t *)sh;                                  // [2 bufs][COLS cols][SB_COL]
    float2 *sBxds = (float2 *)(sh + 2 * COLS * SB_COL);            // [2 bufs][COLS cols][8]
    int8_t *sA    = (int8_t *)(sh + 2 * COLS * SB_COL + 2 * COLS * SBX * 8); // [warps][2 tiles][2 halves][16][32]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    // warps per CTA is a LAUNCH choice: small matmuls (k/v projections) need
    // small CTAs so the grid still covers the SMs. Each warp owns TWO 16-row
    // tiles: the B fragments and xds scales are loaded once and feed both
    // tiles' mma — the profiler put 40% of issue cycles on the MIO queue, and
    // halving the shared traffic per mma is the structural answer.
    const int r0 = blockIdx.x * (32 * (blockDim.x >> 5)) + warp * 32;
    const int gid = lane >> 2, tid = lane & 3;
    const int nbk = k / 256;
    int8_t *sAw = sA + warp * 2 * 2 * 16 * SA_ROW;
    float acc[2][COLS / 8][4];                                     // [tile][colgroup][slot]
    #pragma unroll
    for (int t = 0; t < 2; t++) for (int h = 0; h < COLS / 8; h++) for (int s = 0; s < 4; s++) acc[t][h][s] = 0.0f;

    // this lane's header row: lanes 0..15 hold tile 0's rows, 16..31 tile 1's.
    // in-place uses the clamped row directly; REPACK maps it to (tile-block,
    // row-in-tile) — lanes 16-31 land on tile-block tbrow+1, kept in bounds by
    // the even allocation (the repack already clamped padding rows to m-1).
    int rowL = r0 + lane; if (rowL >= m) rowL = m - 1;
    const unsigned char *myrow = wbase + (size_t)rowL * nbk * ts;
    const int tbrow = r0 / RPACK, htile = (r0 + lane) >> 4, hrr = (r0 + lane) & 15;

    // split-K: blockIdx.z selects a k-slice [kb0,kbe); gridDim.z==1 -> [0,nbk),
    // the single-launch path that is byte-identical to before. Grid-starved
    // small-m matmuls launch z=S extra row-tile CTAs so the 8 SMs fill, each
    // writing a partial to pacc that a combine pass sums. The dispatch picks S
    // so every slice is non-empty (kb0 < nbk always).
    int per = (nbk + (int)gridDim.z - 1) / (int)gridDim.z;
    int kb0 = (int)blockIdx.z * per, kbe = nbk < kb0 + per ? nbk : kb0 + per;
    mma_stage_b<COLS>(sB, sBxds, 0, xq, xds, k, kb0, threadIdx.x);  // prologue: slice's first block
    asm volatile("cp.async.commit_group;\n");

    for (int blk = kb0; blk < kbe; blk++) {
        int buf = (blk - kb0) & 1;
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();                                           // staged data visible CTA-wide
        if (blk + 1 < kbe) {                                       // next block overlaps this one's mma
            mma_stage_b<COLS>(sB, sBxds, buf ^ 1, xq, xds, k, blk + 1, threadIdx.x);
            asm volatile("cp.async.commit_group;\n");
        }
        const int8_t *sBb = sB + buf * COLS * SB_COL;
        const float2 *sBxd = sBxds + buf * COLS * SBX;

        // per-row scale header. in-place: read the q4_K block + decode on the
        // fly. REPACK: read the pre-decoded sc/m (in the half loop) and the
        // (d,dmin) here, from this lane's tile-block.
        uint32_t s0 = 0, s1 = 0, s2 = 0; const uint8_t *sd = NULL;
        float dD, dM;
        if (REPACK) {
            const unsigned char *sctb = wbase + ((size_t)htile * nbk + blk) * Q4KR_TBLK;
            sd = sctb + Q4KR_QUANT;
            const uint16_t *dd = (const uint16_t *)(sctb + Q4KR_QUANT + Q4KR_SCALE);
            dD = d_fp16(dd[hrr * 2 + 0]); dM = d_fp16(dd[hrr * 2 + 1]);
        } else {
            const block_q4_K *hp = (const block_q4_K *)(myrow + (size_t)blk * ts);
            s0 = ((const uint32_t *)hp->scales)[0];
            s1 = ((const uint32_t *)hp->scales)[1];
            s2 = ((const uint32_t *)hp->scales)[2];
            d_dm(&hp->d, &dD, &dM);
        }

        for (int sjp = 0; sjp < 4; sjp++) {                        // sub-blocks in lo/hi PAIRS:
            // they share the same 32 qs bytes, so one load pass stages both
            // unpacked halves, both tiles.
            #pragma unroll
            for (int t = 0; t < 2; t++) {
                int ar = lane & 15, hi = lane >> 4;                // row-in-tile, which 16B half
                uint4 q4;
                if (REPACK) {                                      // coalesced: consecutive lanes contiguous
                    const uint4 *qd = (const uint4 *)(wbase + ((size_t)(tbrow + t) * nbk + blk) * Q4KR_TBLK);
                    q4 = qd[(sjp * 2 + hi) * RPACK + ar];          // group sjp*2+hi, row-in-tile ar (repack tail-clamped)
                } else {
                    int qoff = sjp * 32;
                    int row = r0 + t * 16 + ar; if (row >= m) row = m - 1;
                    const block_q4_K *bp = (const block_q4_K *)(wbase + (size_t)row * nbk * ts) + blk;
                    q4 = *(const uint4 *)(bp->qs + qoff + hi * 16);
                }
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
                uint8_t sc, mm;
                if (REPACK) { sc = sd[hrr * 16 + sj]; mm = sd[hrr * 16 + 8 + sj]; }
                else        { d_gsm32r(sj, s0, s1, s2, &sc, &mm); }
                float dscL = dD * sc, mnmL = dM * mm;              // my header ROW's pair
                float dsc[2][2], mnm[2][2];                        // [tile][row-half]
                #pragma unroll
                for (int t = 0; t < 2; t++) {
                    dsc[t][0] = __shfl_sync(0xffffffffu, dscL, t * 16 + gid);
                    dsc[t][1] = __shfl_sync(0xffffffffu, dscL, t * 16 + gid + 8);
                    mnm[t][0] = __shfl_sync(0xffffffffu, mnmL, t * 16 + gid);
                    mnm[t][1] = __shfl_sync(0xffffffffu, mnmL, t * 16 + gid + 8);
                }
                #pragma unroll
                for (int h = 0; h < COLS / 8; h++) {
                    int colb = h * 8 + gid;
                    uint32_t b0 = *(uint32_t *)(sBb + colb * SB_COL + sj * 32 + tid * 4);
                    uint32_t b1 = *(uint32_t *)(sBb + colb * SB_COL + sj * 32 + tid * 4 + 16);
                    float2 xd0 = sBxd[(h * 8 + tid * 2) * SBX + sj];
                    float2 xd1 = sBxd[(h * 8 + tid * 2 + 1) * SBX + sj];
                    #pragma unroll
                    for (int t = 0; t < 2; t++) {                  // both tiles ride one B load
                        const int8_t *sAh = sAw + t * 2 * 16 * SA_ROW + half * 16 * SA_ROW;
                        uint32_t a0 = *(uint32_t *)(sAh + gid * SA_ROW + tid * 4);
                        uint32_t a1 = *(uint32_t *)(sAh + (gid + 8) * SA_ROW + tid * 4);
                        uint32_t a2 = *(uint32_t *)(sAh + gid * SA_ROW + tid * 4 + 16);
                        uint32_t a3 = *(uint32_t *)(sAh + (gid + 8) * SA_ROW + tid * 4 + 16);
                        int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                            : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                        acc[t][h][0] += xd0.x * (dsc[t][0] * (float)c0 - mnm[t][0] * xd0.y);
                        acc[t][h][1] += xd1.x * (dsc[t][0] * (float)c1 - mnm[t][0] * xd1.y);
                        acc[t][h][2] += xd0.x * (dsc[t][1] * (float)c2 - mnm[t][1] * xd0.y);
                        acc[t][h][3] += xd1.x * (dsc[t][1] * (float)c3 - mnm[t][1] * xd1.y);
                    }
                }
            }
            __syncwarp();                                          // all reads done before re-stage
        }
    }

    // single launch -> out; split -> this slice's partial in pacc[blockIdx.z]
    // (a combine pass sums the slices). [kslice][COLS][m], col-major like out.
    float *o = pacc ? pacc + (size_t)blockIdx.z * COLS * m : out;
    #pragma unroll
    for (int t = 0; t < 2; t++)
        for (int h = 0; h < COLS / 8; h++) {
            int col0 = h * 8 + tid * 2;
            int rA = r0 + t * 16 + gid, rB = r0 + t * 16 + gid + 8;
            if (rA < m) { o[(size_t)col0 * m + rA] = acc[t][h][0]; o[(size_t)(col0 + 1) * m + rA] = acc[t][h][1]; }
            if (rB < m) { o[(size_t)col0 * m + rB] = acc[t][h][2]; o[(size_t)(col0 + 1) * m + rB] = acc[t][h][3]; }
        }
}

// Sum the split-K partials [nsplit][n] -> out[n] (n = COLS*m). Ascending-slice
// sum; combined with each slice's ascending-k accumulation this is the full
// sum, reassociated (relaxed class, deterministic) — prefill only.
__global__ static void combine_pf_kernel(float *out, const float *pacc, int nsplit, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0.0f;
    for (int z = 0; z < nsplit; z++) s += pacc[(size_t)z * n + i];
    out[i] = s;
}
// split-K partial scratch (grows; prefill is not graph-captured, so a mid-run
// grow is safe — but keep-and-reuse to avoid per-matmul mallocs).
static float *g_pacc_pf = NULL; static size_t g_pacc_pf_cap = 0;
static float *ensure_pacc_pf(size_t n) {
    if (n > g_pacc_pf_cap) { cudaFree(g_pacc_pf); CUDA_CHECK(cudaMalloc(&g_pacc_pf, n * sizeof(float))); g_pacc_pf_cap = n; }
    return g_pacc_pf;
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

static void matmul_q_n(float *d_out, const struct gguf_tensor *t, const float *d_x, int k, int m) {
    rweight_init_all();
    int blck = ggml_blck_size(t->type), ts;
    const unsigned char *w = rweight(t, &ts);
    if (t->type < 40) g_cov[t->type] += (double)k * m;
    static int no_mma = -1;
    if (no_mma < 0) no_mma = getenv("LG_NO_MMA") != NULL;
    if ((t->type == GGML_TYPE_Q4_K || t->type == GGML_TYPE_Q6_K) && PREFILL_B % 16 == 0 && !no_mma) {
        static int sms = 0;
        if (!sms) { cudaDeviceProp p; cudaGetDeviceProperties(&p, 0); sms = p.multiProcessorCount; }
        int wpc = 8;                                     // warps (32-row stripes) per CTA: shrink
        while (wpc > 1 && (m + 32 * wpc - 1) / (32 * wpc) < 2 * sms) wpc >>= 1;   // until the grid covers the SMs
        int blocks = (m + 32 * wpc - 1) / (32 * wpc);
        if (t->type == GGML_TYPE_Q4_K) {                          // widen to the whole chunk: B columns, one launch
            size_t shm = 2 * PREFILL_B * SB_COL + 2 * PREFILL_B * SBX * sizeof(float2) + (size_t)wpc * 2 * 2 * 16 * SA_ROW;
            const unsigned char *rw = rweight_mmq(t);             // tile-coalesced twin, or NULL (in-place)
            // split-K: a small-m matmul has only m/32 row-tiles -> too few CTAs
            // (warps/SM ~ m/256) to hide the shared-load latency on 8 SMs. Split
            // the k-dim into S CTAs per tile so blocks*S reaches ~4*SMs; each
            // writes a partial, a combine sums them. Big-m (ffn, blocks>=4*SMs)
            // stays S=1 = the unchanged single launch. LG_NO_SPLITK_PF reverts.
            static int no_spf = -1; if (no_spf < 0) no_spf = getenv("LG_NO_SPLITK_PF") != NULL;
            int nbk = k / 256, S = 1;
            if (!no_spf && nbk >= 2 && blocks < 4 * sms) {
                S = (4 * sms + blocks - 1) / blocks;
                if (S > 8) S = 8;
                if (S > nbk) S = nbk;
                int per = (nbk + S - 1) / S;
                while (S > 1 && (S - 1) * per >= nbk) { S--; per = (nbk + S - 1) / S; }
            }
            if (S > 1) {
                float *pacc = ensure_pacc_pf((size_t)S * PREFILL_B * m);
                dim3 g(blocks, 1, S);
                if (rw) matmul_q4k_mma_kernel<PREFILL_B, true ><<<g, 32 * wpc, shm, g_launch>>>(d_out, rw, ts, g_xq, g_xds, k, m, pacc);
                else    matmul_q4k_mma_kernel<PREFILL_B, false><<<g, 32 * wpc, shm, g_launch>>>(d_out, w,  ts, g_xq, g_xds, k, m, pacc);
                combine_pf_kernel<<<gridn(PREFILL_B * m), 256, 0, g_launch>>>(d_out, pacc, S, PREFILL_B * m);
            } else if (rw) matmul_q4k_mma_kernel<PREFILL_B, true ><<<blocks, 32 * wpc, shm, g_launch>>>(d_out, rw, ts, g_xq, g_xds, k, m, NULL);
            else            matmul_q4k_mma_kernel<PREFILL_B, false><<<blocks, 32 * wpc, shm, g_launch>>>(d_out, w,  ts, g_xq, g_xds, k, m, NULL);
        } else {                                                  // q6_K: widen to the whole chunk like q4_K.
            static int q6b16 = -1;                                 // LG_Q6_B16 = the old 16-wide loop (A/B)
            if (q6b16 < 0) q6b16 = getenv("LG_Q6_B16") != NULL;
            if (q6b16) {
                size_t shm = 2 * 16 * SB_COL + 2 * 16 * SBX * sizeof(float2) + (size_t)wpc * 2 * 2048;
                for (int c0 = 0; c0 < PREFILL_B; c0 += 16)
                    matmul_q6k_mma_kernel<16><<<blocks, 32 * wpc, shm, g_launch>>>(
                        d_out + (size_t)c0 * m, w, ts, g_xq + (size_t)c0 * k, g_xds + (size_t)c0 * (k / 32), k, m);
            } else {                                               // q6_K's A tile is bigger (4 KB/warp), so cap
                int wpc6 = wpc;                                    // warps/CTA to keep shm < 48 KB (no carveout)
                while (wpc6 > 1 && 2 * PREFILL_B * SB_COL + 2 * PREFILL_B * SBX * (int)sizeof(float2) + wpc6 * 2 * 2048 > 48 * 1024) wpc6 >>= 1;
                size_t shm = 2 * PREFILL_B * SB_COL + 2 * PREFILL_B * SBX * sizeof(float2) + (size_t)wpc6 * 2 * 2048;
                int blocks6 = (m + 32 * wpc6 - 1) / (32 * wpc6);
                matmul_q6k_mma_kernel<PREFILL_B><<<blocks6, 32 * wpc6, shm, g_launch>>>(d_out, w, ts, g_xq, g_xds, k, m);
            }
        }
        return;
    }
    // f32/bf16 PLE weight with a q6_K twin -> tensor-core mma (prefill only).
    const unsigned char *q6 = no_mma ? NULL : rweight_q6(t);
    if (q6 && PREFILL_B % 16 == 0) {
        static int sms6 = 0;
        if (!sms6) { cudaDeviceProp p; cudaGetDeviceProperties(&p, 0); sms6 = p.multiProcessorCount; }
        int wpc = 8;
        while (wpc > 1 && (m + 32 * wpc - 1) / (32 * wpc) < 2 * sms6) wpc >>= 1;
        while (wpc > 1 && 2 * PREFILL_B * SB_COL + 2 * PREFILL_B * SBX * (int)sizeof(float2) + wpc * 2 * 2048 > 48 * 1024) wpc >>= 1;
        size_t shm = 2 * PREFILL_B * SB_COL + 2 * PREFILL_B * SBX * sizeof(float2) + (size_t)wpc * 2 * 2048;
        int blocks6 = (m + 32 * wpc - 1) / (32 * wpc);
        matmul_q6k_mma_kernel<PREFILL_B><<<blocks6, 32 * wpc, shm, g_launch>>>(d_out, q6, (int)sizeof(block_q6_Kr), g_xq, g_xds, k, m);
        return;
    }
    int rows_per_block = 256 / 32;
    int blocks = (m + rows_per_block - 1) / rows_per_block;
    matmul_i8r_n_kernel<PREFILL_B><<<blocks, 256, 0, g_launch>>>(d_out, w, (int)t->type, ts, blck, d_x, g_xq, g_xds, k, m);
}
static void matmul_q_2(float *d_out, const struct gguf_tensor *t, const float *d_x, int k, int m) {
    rweight_init_all();
    int blck = ggml_blck_size(t->type), ts;
    const unsigned char *w = rweight(t, &ts);
    int blocks = (m + 7) / 8;
    matmul_i8r_n_kernel<2><<<blocks, 256, 0, g_launch>>>(d_out, w, (int)t->type, ts, blck, d_x, g_xq, g_xds, k, m);
}
