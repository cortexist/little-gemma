// CUDA backend, f32-dequant matmul. The whole forward, kv cache, and non-matmul
// kernels are shared (model-cuda.cuh); this file only provides matmul_q.
//
// This is the readable reference: one warp per output row, the lanes cooperate on
// each quantized block, and the dequant is fused straight into the f32 dot. The
// int8 variant (model-cuda-i8r.cu) is the same forward with a faster matmul — diff
// the two files to see exactly where the speedup comes from.

#include "model-cuda.cuh"

// Accumulate one block's contribution to the dot: each of the 32 lanes handles
// elements {lane, lane+32, ...}. Per-block scale factors are computed once; the
// per-element index math is derived from the sequential layouts in quant.c.

__device__ static void dot_block_q4_K(const block_q4_K *p, int lane, const float *xb, float &s) {
    float d = d_fp16(p->d), mn = d_fp16(p->dmin);
    for (int e = lane; e < QK_K; e += 32) {
        int g = e >> 6, pp = e & 63, is = 2 * g + (pp >= 32 ? 1 : 0);
        uint8_t sc, m; d_gsm(is, p->scales, &sc, &m);
        int qi = (g << 5) + (pp & 31);
        int q = (pp < 32) ? (p->qs[qi] & 0xF) : (p->qs[qi] >> 4);
        s += (d * sc * q - mn * m) * xb[e];
    }
}
__device__ static void dot_block_q5_K(const block_q5_K *p, int lane, const float *xb, float &s) {
    float d = d_fp16(p->d), mn = d_fp16(p->dmin);
    for (int e = lane; e < QK_K; e += 32) {
        int g = e >> 6, pp = e & 63, pos = pp & 31, is = 2 * g + (pp >= 32 ? 1 : 0);
        uint8_t sc, m; d_gsm(is, p->scales, &sc, &m);
        int lo = pp < 32, q = lo ? (p->qs[(g << 5) + pos] & 0xF) : (p->qs[(g << 5) + pos] >> 4);
        int bit = lo ? (1 << (2 * g)) : (2 << (2 * g));
        int hi = (p->qh[pos] & bit) ? 16 : 0;
        s += (d * sc * (q + hi) - mn * m) * xb[e];
    }
}
__device__ static void dot_block_q3_K(const block_q3_K *p, int lane, const float *xb, float &s) {
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    uint32_t aux[4]; const int8_t *scales = (const int8_t *)aux;
    memcpy(aux, p->scales, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = ((aux[0] >> 0) & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = ((aux[1] >> 0) & kmask2) | (((tmp >> 2) & kmask1) << 4);
    float d_all = d_fp16(p->d);
    for (int e = lane; e < QK_K; e += 32) {
        int ni = e >> 7, within = e & 127, j = within >> 5, w2 = within & 31, half = w2 >> 4, l = w2 & 15;
        int is = ni * 8 + j * 2 + half, shift = 2 * j;
        uint8_t m = (uint8_t)(1u << (ni * 4 + j));
        int qv = (int)((int8_t)((p->qs[ni * 32 + half * 16 + l] >> shift) & 3)) - ((p->hmask[half * 16 + l] & m) ? 0 : 4);
        s += (d_all * (scales[is] - 32) * qv) * xb[e];
    }
}
__device__ static void dot_block_q6_K(const block_q6_K *p, int lane, const float *xb, float &s) {
    float d = d_fp16(p->d);
    for (int e = lane; e < QK_K; e += 32) {
        int ni = e >> 7, within = e & 127, grp = within >> 5, l = within & 31, is = l >> 4;
        const uint8_t *ql = p->ql + ni * 64, *qh = p->qh + ni * 32;
        const int8_t *sc = p->scales + ni * 8;
        int qlo, sh, scoff;
        if (grp == 0)      { qlo = ql[l] & 0xF;      sh = 0; scoff = 0; }
        else if (grp == 1) { qlo = ql[l + 32] & 0xF; sh = 2; scoff = 2; }
        else if (grp == 2) { qlo = ql[l] >> 4;       sh = 4; scoff = 4; }
        else               { qlo = ql[l + 32] >> 4;  sh = 6; scoff = 6; }
        int q = (int)((int8_t)(qlo | (((qh[l] >> sh) & 3) << 4))) - 32;
        s += (d * sc[is + scoff] * q) * xb[e];
    }
}
__device__ static void dot_block_q8_0(const block_q8_0 *p, int lane, const float *xb, float &s) {
    float d = d_fp16(p->d);
    for (int e = lane; e < 32; e += 32) s += (d * p->qs[e]) * xb[e];
}

// out[i] = W[i,:] . x : one warp per output row; the whole warp cooperates on each
// block and reduces with a shuffle.
__global__ static void matmul_q_kernel(float *out, const unsigned char *wbase,
                                       int type, int ts, int blck, const float *x, int k, int m) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= m) return;
    const unsigned char *row = wbase + (size_t)warp * (k / blck) * ts;
    int nb = k / blck;
    float s = 0.0f;
    for (int b = 0; b < nb; b++) {
        const unsigned char *blk = row + (size_t)b * ts;
        const float *xb = x + (size_t)b * blck;
        switch (type) {
            case GGML_TYPE_Q4_K: dot_block_q4_K((const block_q4_K *)blk, lane, xb, s); break;
            case GGML_TYPE_Q3_K: dot_block_q3_K((const block_q3_K *)blk, lane, xb, s); break;
            case GGML_TYPE_Q5_K: dot_block_q5_K((const block_q5_K *)blk, lane, xb, s); break;
            case GGML_TYPE_Q6_K: dot_block_q6_K((const block_q6_K *)blk, lane, xb, s); break;
            case GGML_TYPE_Q8_0: dot_block_q8_0((const block_q8_0 *)blk, lane, xb, s); break;
            case GGML_TYPE_F32:  if (lane == 0) s += (*(const float *)blk) * xb[0]; break;
            case GGML_TYPE_BF16: if (lane == 0) s += d_bf16(*(const uint16_t *)blk) * xb[0]; break;
            case GGML_TYPE_F16:  if (lane == 0) s += d_fp16(*(const uint16_t *)blk) * xb[0]; break;
        }
    }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if (lane == 0) out[warp] = s;
}

static void matmul_q(float *d_out, const struct gguf_tensor *t, const float *d_x, int k, int m) {
    int blck = ggml_blck_size(t->type), ts = (int)ggml_type_size(t->type);
    int rows_per_block = 256 / 32;                 // 8 warps per block, one row each
    int blocks = (m + rows_per_block - 1) / rows_per_block;
    matmul_q_kernel<<<blocks, 256, 0, g_launch>>>(d_out, dev_weight(t), (int)t->type, ts, blck, d_x, k, m);
}
// The chunk form, as a plain loop: the readable backend keeps the one-column
// kernel and forgoes the weight-reuse win (see model-cuda-i8r.cu for the real
// thing — its batched kernel reads each weight row once for the whole chunk).
static void matmul_q_n(float *d_out, const struct gguf_tensor *t, const float *d_x, int k, int m) {
    for (int j = 0; j < PREFILL_B; j++)
        matmul_q(d_out + (size_t)j * m, t, d_x + (size_t)j * k, k, m);
}
// The f32 backend dots the float activation directly — no quantize epilogues.
static struct actq actq_for(int k) { (void)k; return AQ0; }
static void act_quantize(const float *d_x, int k) { (void)d_x; (void)k; }
