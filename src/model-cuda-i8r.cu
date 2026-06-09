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

__device__ static float sub_q4_K(const block_q4_K *p, int sj, const int8_t *xqb, const float *xdb, const int *xsb) {
    uint8_t sc, mm; d_gsm(sj, p->scales, &sc, &mm);
    int g = sj >> 1, half = sj & 1;
    const uint8_t *q = p->qs + g * 32;
    const int8_t *xqg = xqb + sj * 32;             // sub-block sj == activation group sj
    int dot = 0;
    for (int u = 0; u < 32; u += 4) {
        uint32_t q32 = ld32(q + u);                // 4 bytes = 4 nibbles of each half
        int w = (int)(half ? (q32 >> 4) & 0x0F0F0F0Fu : q32 & 0x0F0F0F0Fu);
        dot = __dp4a(w, *(const int *)(xqg + u), dot);
    }
    float d = d_fp16(p->d), mn = d_fp16(p->dmin);
    return xdb[sj] * (d * sc * dot - mn * mm * xsb[sj]);
}
__device__ static float sub_q5_K(const block_q5_K *p, int sj, const int8_t *xqb, const float *xdb, const int *xsb) {
    uint8_t sc, mm; d_gsm(sj, p->scales, &sc, &mm);
    int g = sj >> 1, half = sj & 1;
    const uint8_t *q = p->qs + g * 32;
    int bitpos = 2 * g + half;                     // which qh bit holds this sub-block's 5th bit
    const int8_t *xqg = xqb + sj * 32;
    int dot = 0;                                   // 4-bit + high bit -> value 0..31
    for (int u = 0; u < 32; u += 4) {
        uint32_t q32 = ld32(q + u), h32 = ld32(p->qh + u);
        uint32_t nib = half ? (q32 >> 4) & 0x0F0F0F0Fu : q32 & 0x0F0F0F0Fu;
        uint32_t hi  = ((h32 >> bitpos) & 0x01010101u) << 4;
        dot = __dp4a((int)(nib | hi), *(const int *)(xqg + u), dot);
    }
    float d = d_fp16(p->d), mn = d_fp16(p->dmin);
    return xdb[sj] * (d * sc * dot - mn * mm * xsb[sj]);
}
__device__ static float sub_q3_Kr(const block_q3_Kr *p, int sj, const int8_t *xqb, const float *xdb) {
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
    return xdb[g_act] * d_fp16(p->d) * p->scales[sj] * dot;
}
__device__ static float sub_q6_Kr(const block_q6_Kr *p, int sj, const int8_t *xqb, const float *xdb) {
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
    return xdb[g_act] * d_fp16(p->d) * p->scales[ni * 8 + 2 * grp + hl] * dot;
}
__device__ static float sub_q8_0(const block_q8_0 *p, const int8_t *xqb, const float *xdb) {
    int dot = 0;                                   // not in the test models; kept scalar
    for (int i = 0; i < 32; i++) dot += (int)p->qs[i] * xqb[i];
    return xdb[0] * d_fp16(p->d) * dot;
}

// out[i] = W[i,:] . x : one warp per output row. For K-quant/q8_0 weights the
// lanes split the row's sub-blocks (integer dots); for bf16/f32/f16 they fall
// back to the f32 path (lane 0). A shuffle reduces the per-lane partials.
// wbase/ts refer to the repacked layout for q3_K/q6_K, the original otherwise.
// (2 rows per warp — an ILP twist on the empirically-dead split-k idea — was
// tried here and also regressed, uniform, adaptive, and template-specialized
// alike; the one-row shape stays.)
__global__ static void matmul_i8r_kernel(float *out, const unsigned char *wbase, int type, int ts, int blck,
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
                case GGML_TYPE_Q3_K: s += sub_q3_Kr((const block_q3_Kr *)blk, sj, xqb, xdb); break;
                case GGML_TYPE_Q6_K: s += sub_q6_Kr((const block_q6_Kr *)blk, sj, xqb, xdb); break;
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

// The int8 matmul against the activation currently in g_xq/g_xd/g_xs. (d_x is still
// needed for the bf16/f32/f16 fallback path, which dots the raw float activation.)
static void matmul_run(float *d_out, const struct gguf_tensor *t, const float *d_x, int k, int m) {
    int blck = ggml_blck_size(t->type), ts;
    const unsigned char *w = rweight(t, &ts);
    int rows_per_block = 256 / 32;
    int blocks = (m + rows_per_block - 1) / rows_per_block;
    matmul_i8r_kernel<<<blocks, 256>>>(d_out, w, (int)t->type, ts, blck, d_x, g_xq, g_xd, g_xs, k, m);
}

static void matmul_q(float *d_out, const struct gguf_tensor *t, const float *d_x, int k, int m) {
    int ng = k / 32;
    ensure_act(k);
    quantize_act_kernel<<<gridn(ng), 256>>>(d_x, g_xq, g_xd, g_xs, ng);
    matmul_run(d_out, t, d_x, k, m);
}

// Reuse the activation quantized by the immediately preceding matmul_q. The forward
// hands the same input vector to q/k/v (and to gate/up), so re-quantizing it each time
// is pure waste (quantize was ~13% of GPU time). Caller guarantees d_x is unchanged
// since that matmul_q; the result is bit-identical to calling matmul_q again.
static void matmul_q_same(float *d_out, const struct gguf_tensor *t, const float *d_x, int k, int m) {
    matmul_run(d_out, t, d_x, k, m);
}
