// CUDA backend, int8 matmul. Same shared forward as model-cuda.cu
// (model-cuda-common.cuh); only matmul_q differs — this is where the gap to
// llama.cpp closes.
//
// Idea (llama.cpp's mul_mat_vec_q, simplified): the activation x is the same for
// every output row, so quantize it to int8 ONCE per matmul — per 32-element group,
// a scale d_x and the int8 values (plus their sum, for weights that carry a min).
// Then the per-row dot is done in INTEGER arithmetic over each weight sub-block,
// and the float scale is applied once per sub-block instead of once per element.
// For a weight value  w = d*sc*q (- dmin*m)  and activation  x ≈ d_x*xq:
//     sum_i w_i x_i = d_x ( d*sc * Σ q_i·xq_i  [- dmin*m * Σ xq_i] )
// The Σ q_i·xq_i is a small-integer dot; everything float happens once per
// sub-block. This is lossy (int8 activation), exactly like llama.cpp's GPU path.

#include "model-cuda-common.cuh"

// Quantize the activation into int8 per 32-element group: xq (int8), xd (scale),
// xs (sum of the int8 values, used for weight min terms). One thread per group.
__global__ static void quantize_act_kernel(const float *x, int8_t *xq, float *xd, int *xs, int ng) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ng) return;
    const float *xb = x + (size_t)g * 32;
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(xb[i]));
    float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
    int8_t *q = xq + (size_t)g * 32;
    int sum = 0;
    for (int i = 0; i < 32; i++) { int v = __float2int_rn(xb[i] * id); q[i] = (int8_t)v; sum += v; }
    xd[g] = d; xs[g] = sum;
}

// ---- one weight sub-block's contribution to the dot, in integer arithmetic ----
// xqb/xdb/xsb point at the activation int8 / scale / sum for this weight block's
// elements (group stride 32). The per-element index math mirrors model-cuda.cu.

__device__ static float sub_q4_K(const block_q4_K *p, int sj, const int8_t *xqb, const float *xdb, const int *xsb) {
    uint8_t sc, mm; d_gsm(sj, p->scales, &sc, &mm);
    int g = sj >> 1, half = sj & 1;
    const uint8_t *q = p->qs + g * 32;
    const int8_t *xqg = xqb + sj * 32;             // sub-block sj == activation group sj
    int dot = 0;
    for (int i = 0; i < 32; i++) dot += (half ? (q[i] >> 4) : (q[i] & 0xF)) * xqg[i];
    float d = d_fp16(p->d), mn = d_fp16(p->dmin);
    return xdb[sj] * (d * sc * dot - mn * mm * xsb[sj]);
}
__device__ static float sub_q5_K(const block_q5_K *p, int sj, const int8_t *xqb, const float *xdb, const int *xsb) {
    uint8_t sc, mm; d_gsm(sj, p->scales, &sc, &mm);
    int g = sj >> 1, half = sj & 1;
    const uint8_t *q = p->qs + g * 32;
    int bit = half ? (2 << (2 * g)) : (1 << (2 * g));
    const int8_t *xqg = xqb + sj * 32;
    int dot = 0;
    for (int i = 0; i < 32; i++) {
        int hi = (p->qh[i] & bit) ? 16 : 0;
        dot += ((half ? (q[i] >> 4) : (q[i] & 0xF)) + hi) * xqg[i];
    }
    float d = d_fp16(p->d), mn = d_fp16(p->dmin);
    return xdb[sj] * (d * sc * dot - mn * mm * xsb[sj]);
}
__device__ static float sub_q3_K(const block_q3_K *p, int sj, const int8_t *xqb, const float *xdb) {
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    uint32_t aux[4]; const int8_t *scales = (const int8_t *)aux;
    memcpy(aux, p->scales, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = ((aux[0] >> 0) & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = ((aux[1] >> 0) & kmask2) | (((tmp >> 2) & kmask1) << 4);
    int ni = sj >> 3, rem = sj & 7, j = rem >> 1, half = rem & 1;
    int shift = 2 * j; uint8_t m = (uint8_t)(1u << (ni * 4 + j));
    const uint8_t *qs = p->qs + ni * 32;
    int g_act = ni * 4 + j;
    const int8_t *xqg = xqb + g_act * 32 + half * 16;
    int dot = 0;
    for (int l = 0; l < 16; l++) {
        int qv = (int)((int8_t)((qs[half * 16 + l] >> shift) & 3)) - ((p->hmask[half * 16 + l] & m) ? 0 : 4);
        dot += qv * xqg[l];
    }
    return xdb[g_act] * d_fp16(p->d) * (scales[sj] - 32) * dot;
}
__device__ static float sub_q6_K(const block_q6_K *p, int sj, const int8_t *xqb, const float *xdb) {
    int ni = sj >> 3, rem = sj & 7, grp = rem >> 1, hl = rem & 1;
    const uint8_t *ql = p->ql + ni * 64, *qh = p->qh + ni * 32;
    const int8_t *sc = p->scales + ni * 8;
    int sh = grp * 2, off = (grp & 1) ? 32 : 0, hin = (grp >= 2);
    int g_act = ni * 4 + grp;
    const int8_t *xqg = xqb + g_act * 32 + hl * 16;
    int dot = 0;
    for (int i = 0; i < 16; i++) {
        int l = hl * 16 + i;
        int qlo = hin ? (ql[l + off] >> 4) : (ql[l + off] & 0xF);
        int qval = (int)((int8_t)(qlo | (((qh[l] >> sh) & 3) << 4))) - 32;
        dot += qval * xqg[i];
    }
    return xdb[g_act] * d_fp16(p->d) * sc[2 * grp + hl] * dot;
}
__device__ static float sub_q8_0(const block_q8_0 *p, const int8_t *xqb, const float *xdb) {
    int dot = 0;
    for (int i = 0; i < 32; i++) dot += (int)p->qs[i] * xqb[i];
    return xdb[0] * d_fp16(p->d) * dot;
}

// out[i] = W[i,:] . x : one warp per output row. For K-quant/q8_0 weights the
// lanes split the row's sub-blocks (integer dots); for bf16/f32/f16 they fall back
// to the f32 path (lane 0). A shuffle reduces the per-lane partials.
__global__ static void matmul_i8_kernel(float *out, const unsigned char *wbase, int type, int ts, int blck,
                                        const float *x, const int8_t *xq, const float *xd, const int *xs, int k, int m) {
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
            const float  *xdb = xd + (size_t)b * gpb;
            const int    *xsb = xs + (size_t)b * gpb;
            switch (type) {
                case GGML_TYPE_Q4_K: s += sub_q4_K((const block_q4_K *)blk, sj, xqb, xdb, xsb); break;
                case GGML_TYPE_Q5_K: s += sub_q5_K((const block_q5_K *)blk, sj, xqb, xdb, xsb); break;
                case GGML_TYPE_Q3_K: s += sub_q3_K((const block_q3_K *)blk, sj, xqb, xdb); break;
                case GGML_TYPE_Q6_K: s += sub_q6_K((const block_q6_K *)blk, sj, xqb, xdb); break;
                case GGML_TYPE_Q8_0: s += sub_q8_0((const block_q8_0 *)blk, xqb, xdb); break;
            }
        }
    } else if (lane == 0) {                              // bf16 / f32 / f16 fallback
        for (int b = 0; b < nb; b++) {
            const unsigned char *blk = row + (size_t)b * ts;
            float xb0 = x[(size_t)b * blck];
            if (type == GGML_TYPE_F32)       s += (*(const float *)blk) * xb0;
            else if (type == GGML_TYPE_BF16) s += d_bf16(*(const uint16_t *)blk) * xb0;
            else if (type == GGML_TYPE_F16)  s += d_fp16(*(const uint16_t *)blk) * xb0;
        }
    }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if (lane == 0) out[warp] = s;
}

// Quantized activation scratch (grows as needed; reused across matmuls).
static int8_t *g_xq = NULL; static float *g_xd = NULL; static int *g_xs = NULL;
static int g_act_cap = 0;
static void ensure_act(int k) {
    if (k <= g_act_cap) return;
    cudaFree(g_xq); cudaFree(g_xd); cudaFree(g_xs);
    CUDA_CHECK(cudaMalloc(&g_xq, (size_t)k));
    CUDA_CHECK(cudaMalloc(&g_xd, (size_t)(k / 32) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_xs, (size_t)(k / 32) * sizeof(int)));
    g_act_cap = k;
}

static void matmul_q(float *d_out, const struct gguf_tensor *t, const float *d_x, int k, int m) {
    int blck = ggml_blck_size(t->type), ts = (int)ggml_type_size(t->type);
    int ng = k / 32;
    ensure_act(k);
    quantize_act_kernel<<<gridn(ng), 256>>>(d_x, g_xq, g_xd, g_xs, ng);
    int rows_per_block = 256 / 32;
    int blocks = (m + rows_per_block - 1) / rows_per_block;
    matmul_i8_kernel<<<blocks, 256>>>(d_out, dev_weight(t), (int)t->type, ts, blck, d_x, g_xq, g_xd, g_xs, k, m);
}
