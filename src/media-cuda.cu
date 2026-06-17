// GPU gemma4v encoder — the same 16-block vision transformer media.c runs on
// the host, as a handful of plain CUDA kernels. The host path stays in the
// binary as oracle (LG_MEDIA_VERIFY=1) and fallback; this file exists because
// even a well-vectorized CPU needs ~15 s per 266-token image for the ~1 TFLOP
// this costs, and an A5000 does it in well under a second.
//
// Everything is f32 with the f16 weights converted in the GEMM's inner loop —
// no quantization, no CUDA graph, default stream: the encoder runs once per
// image, latency hiding is not the game here, readability is.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <cuda_fp16.h>

extern "C" {                     // the C headers carry no C++ linkage guards
#include "gguf.h"
#include "quant.h"
#include "media-internal.h"
}

#define VC_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "media-cuda: %s -> %s\n", #x, cudaGetErrorString(e_)); return NULL; } } while (0)

// Device-side mirror of struct vlayer.
struct vld {
    float *ln1, *ln2, *attn_post, *ffn_post, *qn, *kn;
    __half *q, *k, *v, *o, *gate, *up, *down;
};

struct vcuda {
    struct vld *vl;                 // host array of device pointers
    __half *patch16, *mmv;
    int np_cap;                     // current size of the work buffers, in patches
    float *F, *posadd;              // [np][768] im2col input, position-table sum
    float *X, *H, *Q, *K, *V, *D;   // [np][768]
    float *G, *U;                   // [np][3072]
    float *pooled, *rows;           // [np/9][768], [np/9][n_embd]
};

// ---- kernels ----------------------------------------------------------------

// C[t][n] = A[t][k] . W[n][k] — 64x64 register-tiled GEMM: 16x16 threads, each
// holding a 4x4 patch of outputs in registers. Two wins over the plain 16x16
// tiling this replaced: every weight slab is read from global memory once per
// 64 positions instead of per 16 (the whole encoder, ~300MB, was re-streamed
// once per tile row), and the inner loop's 8 shared loads feed 16 FMAs where
// the old loop paid 2 loads per FMA. The k-slabs are staged k-major so a
// thread's 4 a-values and 4 w-values are each one float4 read. Each output
// still accumulates in plain ascending-k order: same floats, same order,
// byte-identical results — only the thread-to-output mapping changed.
#define TS 16                   // k-slab depth, and threads per block side
#define BT 64                   // output tile side (4 outputs per thread per side)
static __global__ void k_gemm(float *C, const float *A, const __half *W, int T, int K, int N) {
    __shared__ float a[TS][BT + 4], w[TS][BT + 4];      // +4: float4-aligned, conflict-free
    int t0 = blockIdx.y * BT, n0 = blockIdx.x * BT;
    int tid = threadIdx.y * TS + threadIdx.x;
    float acc[4][4] = { 0 };
    for (int k0 = 0; k0 < K; k0 += TS) {
        for (int e = tid; e < BT * TS; e += TS * TS) {  // stage both slabs, k-major
            int r = e >> 4, i = e & (TS - 1);
            int kk = k0 + i;                            // guard K too: audio's inp-proj /
            a[i][r] = t0 + r < T && kk < K ? A[(size_t)(t0 + r) * K + kk] : 0.0f;   // mm_a widths
            w[i][r] = n0 + r < N && kk < K ? __half2float(W[(size_t)(n0 + r) * K + kk]) : 0.0f;  // need not be /16
        }
        __syncthreads();
        for (int i = 0; i < TS; i++) {
            float av[4], wv[4];
            *(float4 *)av = *(const float4 *)&a[i][threadIdx.y * 4];
            *(float4 *)wv = *(const float4 *)&w[i][threadIdx.x * 4];
            for (int y = 0; y < 4; y++)
                for (int x = 0; x < 4; x++) acc[y][x] += av[y] * wv[x];
        }
        __syncthreads();
    }
    for (int y = 0; y < 4; y++) {
        int tt = t0 + threadIdx.y * 4 + y;
        if (tt >= T) break;
        for (int x = 0; x < 4; x++) {
            int tn = n0 + threadIdx.x * 4 + x;
            if (tn < N) C[(size_t)tt * N + tn] = acc[y][x];
        }
    }
}

// out[row] = rms_norm(in[row]) * w   (w NULL = plain norm). One block per row.
static __global__ void k_rms(float *out, const float *in, const float *w, int n, float eps) {
    __shared__ float red[256];
    const float *x = in + (size_t)blockIdx.x * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += 256) ss += x[i] * x[i];
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float sc = rsqrtf(red[0] / (float)n + eps);
    float *y = out + (size_t)blockIdx.x * n;
    for (int i = threadIdx.x; i < n; i += 256) y[i] = x[i] * sc * (w ? w[i] : 1.0f);
}

static __global__ void k_add(float *x, const float *a, size_t total) {
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < total; i += gridDim.x * 256ull) x[i] += a[i];
}

static __global__ void k_gelu_mul(float *g, const float *u, size_t total) {
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < total; i += gridDim.x * 256ull) {
        float x = g[i];
        g[i] = x / (1.0f + __expf(-1.702f * x)) * u[i];
    }
}

// rms-norm one 64-wide head segment with a warp (2 lanes-worth of elements each).
static __device__ void d_rms64(float *s, const float *w, float eps, int lane) {
    float ss = 0.0f;
    for (int i = lane; i < 64; i += 32) ss += s[i] * s[i];
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    float sc = rsqrtf(ss / 64.0f + eps);
    for (int i = lane; i < 64; i += 32) s[i] *= sc * (w ? w[i & 63] : 1.0f);
}

// per-head q/k norms (weighted) + plain norm on V. Block = position, warp = head.
static __global__ void k_headnorm(float *Q, float *K, float *V, const float *qn, const float *kn,
                                  int ne, float eps) {
    int p = blockIdx.x, hh = threadIdx.x >> 5, lane = threadIdx.x & 31;
    size_t at = (size_t)p * ne + hh * 64;
    d_rms64(Q + at, qn, eps, lane);
    d_rms64(K + at, kn, eps, lane);
    d_rms64(V + at, NULL, eps, lane);
}

// 2D NeoX RoPE: first 32 dims of each head rotate by patch column, second 32
// by row; NeoX pairs (i, i+16) inside each half. Block = position, warp = head.
static __global__ void k_rope2d(float *X, int ne, int n_cols, float theta) {
    int p = blockIdx.x, hh = threadIdx.x >> 5, r = threadIdx.x & 31;
    int sec = r >> 4, i = r & 15;
    int pos = sec ? p / n_cols : p % n_cols;
    float freq = __powf(theta, -2.0f * (float)i / 32.0f);
    float ang = (float)pos * freq, c = __cosf(ang), sn = __sinf(ang);
    float *s = X + (size_t)p * ne + hh * 64 + sec * 32;
    float a = s[i], b = s[i + 16];
    s[i]      = a * c - b * sn;
    s[i + 16] = a * sn + b * c;
}

// Full attention at scale 1.0. Block = (query, head); shmem holds the np
// scores + a 256-wide reduction scratch. Scores: warp per key row, lanes split
// the 64-dim dot. AV: thread (i, slice) so V reads coalesce over i.
static __global__ void k_attn(float *D, const float *Q, const float *K, const float *V,
                              int np, int ne) {
    extern __shared__ float sm[];
    float *att = sm, *red = sm + np;
    int p = blockIdx.x, hh = blockIdx.y;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float *q = Q + (size_t)p * ne + hh * 64;
    for (int kk = warp; kk < np; kk += 8) {
        const float *kr = K + (size_t)kk * ne + hh * 64;
        float s = q[lane] * kr[lane] + q[lane + 32] * kr[lane + 32];
        for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
        if (!lane) att[kk] = s;
    }
    __syncthreads();
    float mx = -1e30f;
    for (int kk = threadIdx.x; kk < np; kk += 256) mx = fmaxf(mx, att[kk]);
    red[threadIdx.x] = mx;
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    mx = red[0];
    __syncthreads();
    float sum = 0.0f;
    for (int kk = threadIdx.x; kk < np; kk += 256) { float e = __expf(att[kk] - mx); att[kk] = e; sum += e; }
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float inv = 1.0f / red[0];
    __syncthreads();
    int i = threadIdx.x & 63, sl = threadIdx.x >> 6;    // 4 slices over the keys
    float o = 0.0f;
    for (int kk = sl; kk < np; kk += 4) o += att[kk] * V[(size_t)kk * ne + hh * 64 + i];
    red[threadIdx.x] = o;
    __syncthreads();
    if (!sl) D[(size_t)p * ne + hh * 64 + i] = (red[i] + red[64 + i] + red[128 + i] + red[192 + i]) * inv;
}

// 3x3 average pool over the patch grid, x sqrt(768). Block = output token.
static __global__ void k_pool(float *out, const float *X, int n_cols, int out_x, int ne, float sc) {
    int tx = blockIdx.x % out_x, ty = blockIdx.x / out_x;
    for (int i = threadIdx.x; i < ne; i += 256) {
        float s = 0.0f;
        for (int dy = 0; dy < 3; dy++)
            for (int dx = 0; dx < 3; dx++)
                s += X[(size_t)((ty * 3 + dy) * n_cols + (tx * 3 + dx)) * ne + i];
        out[(size_t)blockIdx.x * ne + i] = s * sc / 9.0f;
    }
}

// ---- upload / init ----------------------------------------------------------

static __half *up_h(const struct gguf_tensor *t) {     // weight -> device f16
    size_t n = 1;
    for (uint32_t i = 0; i < t->n_dims; i++) n *= t->dims[i];
    __half *d = NULL;
    if (cudaMalloc(&d, n * 2) != cudaSuccess) return NULL;
    if (t->type == GGML_TYPE_F16) {
        if (cudaMemcpy(d, t->data, n * 2, cudaMemcpyHostToDevice) != cudaSuccess) return NULL;
    } else {                                            // f32 (or any) -> f16 on host
        float *f = (float *)malloc(n * 4);
        __half *h = (__half *)malloc(n * 2);
        if (!f || !h || !dequantize_into(t->type, t->data, f, (int64_t)n)) { free(f); free(h); return NULL; }
        for (size_t i = 0; i < n; i++) h[i] = __float2half(f[i]);
        cudaError_t e = cudaMemcpy(d, h, n * 2, cudaMemcpyHostToDevice);
        free(f); free(h);
        if (e != cudaSuccess) return NULL;
    }
    return d;
}

static float *up_f(const float *src, size_t n) {       // norm weights -> device f32
    float *d = NULL;
    if (cudaMalloc(&d, n * 4) != cudaSuccess) return NULL;
    if (cudaMemcpy(d, src, n * 4, cudaMemcpyHostToDevice) != cudaSuccess) return NULL;
    return d;
}

static struct vcuda *vcuda_init(struct media *md) {
    if (md->v_embd / md->v_head != 64) {                // kernels assume dh = 64
        fprintf(stderr, "media-cuda: head dim %d != 64, using the host encoder\n", md->v_embd / md->v_head);
        return NULL;
    }
    struct vcuda *vc = (struct vcuda *)calloc(1, sizeof *vc);
    if (!vc) return NULL;
    vc->vl = (struct vld *)calloc((size_t)md->v_layer, sizeof *vc->vl);
    vc->patch16 = up_h(md->v_patch16);
    vc->mmv = up_h(md->mm_v);
    int ok = vc->vl && vc->patch16 && vc->mmv;
    int ne = md->v_embd, dh = 64;
    for (int L = 0; ok && L < md->v_layer; L++) {
        const struct vlayer *s = &md->vl[L];
        struct vld *d = &vc->vl[L];
        ok = (d->ln1 = up_f(s->ln1, ne)) && (d->ln2 = up_f(s->ln2, ne)) &&
             (d->attn_post = up_f(s->attn_post, ne)) && (d->ffn_post = up_f(s->ffn_post, ne)) &&
             (d->qn = up_f(s->qn, dh)) && (d->kn = up_f(s->kn, dh)) &&
             (d->q = up_h(s->q)) && (d->k = up_h(s->k)) && (d->v = up_h(s->v)) &&
             (d->o = up_h(s->o)) && (d->gate = up_h(s->gate)) &&
             (d->up = up_h(s->up)) && (d->down = up_h(s->down)) != NULL;
    }
    if (!ok) {
        fprintf(stderr, "media-cuda: weight upload failed, using the host encoder\n");
        free(vc->vl); free(vc);                        // leaked partial uploads are acceptable here:
        return NULL;                                   // this only happens out-of-memory at startup
    }
    return vc;
}

static int ensure_bufs(struct vcuda *vc, int np, int ne, int n_embd) {
    if (np <= vc->np_cap) return 0;
    float **bufs[] = { &vc->F, &vc->posadd, &vc->X, &vc->H, &vc->Q, &vc->K, &vc->V, &vc->D,
                       &vc->G, &vc->U, &vc->pooled, &vc->rows };
    size_t sz[] = { (size_t)np * ne, (size_t)np * ne, (size_t)np * ne, (size_t)np * ne,
                    (size_t)np * ne, (size_t)np * ne, (size_t)np * ne, (size_t)np * ne,
                    (size_t)np * 4 * ne, (size_t)np * 4 * ne,
                    (size_t)(np / 9 + 1) * ne, (size_t)(np / 9 + 1) * n_embd };
    for (int i = 0; i < 12; i++) {
        cudaFree(*bufs[i]);
        if (cudaMalloc(bufs[i], sz[i] * 4) != cudaSuccess) { vc->np_cap = 0; return -1; }
    }
    vc->np_cap = np;
    return 0;
}

// ---- the encoder ------------------------------------------------------------

extern "C" float *v_embed_image_cuda(struct media *md, const uint8_t *rgb, int w, int h, int *n_tokens) {
    if (md->cuda == (void *)-1) return NULL;            // init already failed once
    struct vcuda *vc = (struct vcuda *)md->cuda;
    if (!vc) {
        vc = vcuda_init(md);
        md->cuda = vc ? (void *)vc : (void *)-1;
        if (!vc) return NULL;
    }

    const int P = md->patch, ne = md->v_embd, nh = md->v_head, mg = md->v_merge;
    const float eps = 1e-6f;
    int n_cols = w / P, n_rows = h / P, np = n_cols * n_rows;
    if (n_cols > md->pos_size || n_rows > md->pos_size) {
        fprintf(stderr, "media: image %dx%d exceeds the %d-entry position table\n", w, h, md->pos_size);
        return NULL;
    }
    size_t attn_shmem = (size_t)np * 4 + 1024;
    if (attn_shmem > 48 * 1024 || ensure_bufs(vc, np, ne, md->n_embd) != 0) return NULL;

    // im2col + the two position tables, on the host (cheap), then one upload each
    float *F = (float *)malloc((size_t)np * ne * 4);
    float *pa = (float *)malloc((size_t)np * ne * 4);
    if (!F || !pa) { free(F); free(pa); return NULL; }
    for (int p = 0; p < np; p++) {
        int col = p % n_cols, row = p / n_cols;
        float *vec = F + (size_t)p * ne;
        for (int c = 0; c < 3; c++)
            for (int ky = 0; ky < P; ky++)
                for (int kx = 0; kx < P; kx++)
                    vec[(c * P + ky) * P + kx] =
                        (float)rgb[3 * ((size_t)(row * P + ky) * w + (col * P + kx)) + c] / 255.0f * 2.0f - 1.0f;
        const float *ex = md->v_pos + (size_t)col * ne;
        const float *ey = md->v_pos + ((size_t)md->pos_size + row) * ne;
        for (int i = 0; i < ne; i++) pa[(size_t)p * ne + i] = ex[i] + ey[i];
    }
    cudaMemcpy(vc->F, F, (size_t)np * ne * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(vc->posadd, pa, (size_t)np * ne * 4, cudaMemcpyHostToDevice);
    free(F); free(pa);

    dim3 blk(TS, TS);
    dim3 g_sq((ne + BT - 1) / BT, (np + BT - 1) / BT);          // [np][768] outputs
    dim3 g_up((4 * ne + BT - 1) / BT, (np + BT - 1) / BT);      // [np][3072]
    int nadd = (int)(((size_t)np * ne + 255) / 256);

    k_gemm<<<g_sq, blk>>>(vc->X, vc->F, vc->patch16, np, ne, ne);
    k_add<<<nadd, 256>>>(vc->X, vc->posadd, (size_t)np * ne);

    for (int L = 0; L < md->v_layer; L++) {
        const struct vld *v = &vc->vl[L];
        k_rms<<<np, 256>>>(vc->H, vc->X, v->ln1, ne, eps);
        k_gemm<<<g_sq, blk>>>(vc->Q, vc->H, v->q, np, ne, ne);
        k_gemm<<<g_sq, blk>>>(vc->K, vc->H, v->k, np, ne, ne);
        k_gemm<<<g_sq, blk>>>(vc->V, vc->H, v->v, np, ne, ne);
        k_headnorm<<<np, nh * 32>>>(vc->Q, vc->K, vc->V, v->qn, v->kn, ne, eps);
        k_rope2d<<<np, nh * 32>>>(vc->Q, ne, n_cols, 100.0f);
        k_rope2d<<<np, nh * 32>>>(vc->K, ne, n_cols, 100.0f);
        k_attn<<<dim3(np, nh), 256, attn_shmem>>>(vc->D, vc->Q, vc->K, vc->V, np, ne);
        k_gemm<<<g_sq, blk>>>(vc->H, vc->D, v->o, np, ne, ne);
        k_rms<<<np, 256>>>(vc->H, vc->H, v->attn_post, ne, eps);
        k_add<<<nadd, 256>>>(vc->X, vc->H, (size_t)np * ne);
        k_rms<<<np, 256>>>(vc->H, vc->X, v->ln2, ne, eps);
        k_gemm<<<g_up, blk>>>(vc->G, vc->H, v->gate, np, ne, 4 * ne);
        k_gemm<<<g_up, blk>>>(vc->U, vc->H, v->up, np, ne, 4 * ne);
        k_gelu_mul<<<nadd * 4, 256>>>(vc->G, vc->U, (size_t)np * 4 * ne);
        k_gemm<<<g_sq, blk>>>(vc->D, vc->G, v->down, np, 4 * ne, ne);
        k_rms<<<np, 256>>>(vc->D, vc->D, v->ffn_post, ne, eps);
        k_add<<<nadd, 256>>>(vc->X, vc->D, (size_t)np * ne);
    }

    int out_x = n_cols / mg, out_y = n_rows / mg, n = out_x * out_y;
    k_pool<<<n, 256>>>(vc->pooled, vc->X, n_cols, out_x, ne, sqrtf((float)ne));
    k_gemm<<<dim3((md->n_embd + BT - 1) / BT, (n + BT - 1) / BT), blk>>>
        (vc->rows, vc->pooled, vc->mmv, n, ne, md->n_embd);
    k_rms<<<n, 256>>>(vc->rows, vc->rows, NULL, md->n_embd, eps);

    float *out = (float *)malloc((size_t)n * md->n_embd * 4);
    if (!out) return NULL;
    VC_CHECK(cudaMemcpy(out, vc->rows, (size_t)n * md->n_embd * 4, cudaMemcpyDeviceToHost));
    VC_CHECK(cudaGetLastError());
    *n_tokens = n;
    return out;
}

extern "C" void v_cuda_free(struct media *md) {
    struct vcuda *vc = (struct vcuda *)md->cuda;
    if (!vc || md->cuda == (void *)-1) return;
    for (int L = 0; L < md->v_layer; L++) {
        struct vld *d = &vc->vl[L];
        cudaFree(d->ln1); cudaFree(d->ln2); cudaFree(d->attn_post); cudaFree(d->ffn_post);
        cudaFree(d->qn); cudaFree(d->kn);
        cudaFree(d->q); cudaFree(d->k); cudaFree(d->v); cudaFree(d->o);
        cudaFree(d->gate); cudaFree(d->up); cudaFree(d->down);
    }
    cudaFree(vc->patch16); cudaFree(vc->mmv);
    cudaFree(vc->F); cudaFree(vc->posadd); cudaFree(vc->X); cudaFree(vc->H);
    cudaFree(vc->Q); cudaFree(vc->K); cudaFree(vc->V); cudaFree(vc->D);
    cudaFree(vc->G); cudaFree(vc->U); cudaFree(vc->pooled); cudaFree(vc->rows);
    free(vc->vl);
    free(vc);
    md->cuda = NULL;
}

// ===== gemma4a audio conformer (GPU) =========================================
// The 12-block conformer that media.c runs on the host (a_blocks_host), as CUDA
// kernels. The front end (log-mel + the two subsampling convs) stays on the
// host and hands us F[T][n_feat]; here we project in, run the stack, and project
// out to the LLM width. Same f32-compute / f16-weight, no-graph, default-stream
// style as the vision encoder above — it runs once per clip, so readability wins
// over latency hiding. The host a_blocks_host stays in as the LG_MEDIA_VERIFY
// oracle (the convs/mel it shares are identical; only the blocks are compared).

struct ald {                                  // device mirror of struct alayer
    float *ffn_norm, *ffn_post, *ffn_norm1, *ffn_post1;
    float *attn_pre, *attn_post, *pds;        // pds [dh]
    float *norm_conv, *conv_norm, *ln2, *dw;  // dw [ne][5]
    __half *ffn_up, *ffn_down, *ffn_up1, *ffn_down1;
    __half *q, *k, *v, *o, *k_rel, *pw1, *pw2;
};

struct acuda {
    struct ald *al;
    __half *inp_proj, *out_proj, *mm_a;
    float *out_proj_b, *rpe;                  // out_proj_b [ao], rpe [13][ne]
    int ne, nh, dh, n_layer, n_feat, ao, n_embd;
    int t_cap;                                // size of the work buffers, in frames
    float *F, *X, *H, *G, *D, *Q, *K, *V, *P, *Y, *rows;
};

// ---- audio kernels (k_gemm / k_rms / k_add reused from the vision encoder) ---

static __global__ void k_silu(float *x, size_t total) {
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < total; i += gridDim.x * 256ull) {
        float v = x[i]; x[i] = v / (1.0f + __expf(-v));
    }
}

// x += s * a   (s = 0.5 for the macaron half-step FFNs, 1.0 for full residuals).
static __global__ void k_madd(float *x, const float *a, float s, size_t total) {
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < total; i += gridDim.x * 256ull) x[i] += s * a[i];
}

// Q *= q_scale * per_dim_scale[i % dh];  K *= k_scale.  P (the rel-pos term) is
// left unscaled, exactly as the host does — the score is q_scaled . (k_scaled + p).
static __global__ void k_qk_scale(float *Q, float *K, const float *pds, int dh,
                                  float qs, float ks, size_t total) {
    for (size_t i = blockIdx.x * 256ull + threadIdx.x; i < total; i += gridDim.x * 256ull) {
        Q[i] *= qs * pds[i % dh];
        K[i] *= ks;
    }
}

// GLU: D[t][i] = G[t][i] * sigmoid(G[t][ne + i])   (G is [T][2 ne]).
static __global__ void k_glu(float *D, const float *G, int ne, size_t total) {
    for (size_t idx = blockIdx.x * 256ull + threadIdx.x; idx < total; idx += gridDim.x * 256ull) {
        int i = idx % ne; size_t t = idx / ne;
        const float *g = G + t * 2 * ne;
        D[idx] = g[i] / (1.0f + __expf(-g[ne + i]));
    }
}

// Causal depthwise conv, 5 taps: H[t][i] = sum_j D[t-4+j][i] * dw[i][j], ts >= 0.
static __global__ void k_depthwise(float *H, const float *D, const float *dw, int ne, size_t total) {
    for (size_t idx = blockIdx.x * 256ull + threadIdx.x; idx < total; idx += gridDim.x * 256ull) {
        int i = idx % ne, t = (int)(idx / ne);
        float s = 0.0f;
        for (int j = 0; j < 5; j++) {
            int ts = t - 4 + j;
            if (ts >= 0) s += D[(size_t)ts * ne + i] * dw[(size_t)i * 5 + j];
        }
        H[idx] = s;
    }
}

static __global__ void k_addbias(float *Y, const float *b, int ao, size_t total) {
    for (size_t idx = blockIdx.x * 256ull + threadIdx.x; idx < total; idx += gridDim.x * 256ull)
        Y[idx] += b[idx % ao];
}

// Chunked local attention: each frame t attends to the causal 12-frame window
// [t-11, t]. Block = (frame, head), blockDim.x = dh (a power of 2, <= 1024).
// score = 50*tanh(q.(k+p)/50) over the window, softmax, then the V-weighted sum;
// p is the projected rel-pos row for distance t-gk (table index 12-(t-gk), 12 = self).
static __global__ void k_local_attn(float *Dout, const float *Q, const float *K, const float *V,
                                    const float *P, int ne, int dh) {
    int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    __shared__ float red[1024];
    __shared__ float att[12];
    __shared__ float ssum;
    const float *qh = Q + (size_t)t * ne + h * dh;
    float qv = qh[i];
    int k0 = t - 11 > 0 ? t - 11 : 0, nk = t - k0 + 1;
    for (int g = 0; g < nk; g++) {                     // 12 windowed dot products
        int gk = k0 + g;
        const float *kh = K + (size_t)gk * ne + h * dh;
        const float *ph = P + (size_t)(12 - (t - gk)) * ne + h * dh;
        red[i] = qv * (kh[i] + ph[i]);
        __syncthreads();
        for (int s = dh >> 1; s; s >>= 1) { if (i < s) red[i] += red[i + s]; __syncthreads(); }
        if (i == 0) att[g] = 50.0f * tanhf(red[0] / 50.0f);
        __syncthreads();
    }
    if (i == 0) {                                      // softmax over the <=12 scores
        float mx = -1e30f;
        for (int g = 0; g < nk; g++) if (att[g] > mx) mx = att[g];
        float sum = 0.0f;
        for (int g = 0; g < nk; g++) { att[g] = expf(att[g] - mx); sum += att[g]; }
        ssum = sum;
    }
    __syncthreads();
    float o = 0.0f;
    for (int g = 0; g < nk; g++) o += att[g] * V[(size_t)(k0 + g) * ne + h * dh + i];
    Dout[(size_t)t * ne + h * dh + i] = o / ssum;
}

// ---- audio upload / init ----------------------------------------------------

static struct acuda *acuda_init(struct media *md) {
    int ne = md->a_embd, nh = md->a_head, dh = ne / nh;
    if (dh > 1024 || (dh & (dh - 1)) != 0) {       // the local-attn reduction wants a pow2 dh
        fprintf(stderr, "media-cuda: audio head dim %d unsupported, using the host conformer\n", dh);
        return NULL;
    }
    struct acuda *ac = (struct acuda *)calloc(1, sizeof *ac);
    if (!ac) return NULL;
    ac->al = (struct ald *)calloc((size_t)md->a_layer, sizeof *ac->al);
    ac->ne = ne; ac->nh = nh; ac->dh = dh; ac->n_layer = md->a_layer;
    ac->ao = (int)md->a_out_proj->dims[1];
    ac->n_embd = md->n_embd;
    ac->inp_proj = up_h(md->a_inp_proj);
    ac->out_proj = up_h(md->a_out_proj);
    ac->mm_a     = up_h(md->mm_a);
    ac->out_proj_b = up_f(md->a_out_proj_b, (size_t)ac->ao);
    ac->rpe        = up_f(md->rpe, (size_t)13 * ne);
    int ok = ac->al && ac->inp_proj && ac->out_proj && ac->mm_a && ac->out_proj_b && ac->rpe;
    for (int L = 0; ok && L < md->a_layer; L++) {
        const struct alayer *s = &md->al[L];
        struct ald *d = &ac->al[L];
        ok = (d->ffn_norm = up_f(s->ffn_norm, ne)) && (d->ffn_post = up_f(s->ffn_post, ne)) &&
             (d->ffn_norm1 = up_f(s->ffn_norm1, ne)) && (d->ffn_post1 = up_f(s->ffn_post1, ne)) &&
             (d->attn_pre = up_f(s->attn_pre, ne)) && (d->attn_post = up_f(s->attn_post, ne)) &&
             (d->pds = up_f(s->pds, dh)) && (d->norm_conv = up_f(s->norm_conv, ne)) &&
             (d->conv_norm = up_f(s->conv_norm, ne)) && (d->ln2 = up_f(s->ln2, ne)) &&
             (d->dw = up_f(s->dw, (size_t)ne * 5)) &&
             (d->ffn_up = up_h(s->ffn_up)) && (d->ffn_down = up_h(s->ffn_down)) &&
             (d->ffn_up1 = up_h(s->ffn_up1)) && (d->ffn_down1 = up_h(s->ffn_down1)) &&
             (d->q = up_h(s->q)) && (d->k = up_h(s->k)) && (d->v = up_h(s->v)) &&
             (d->o = up_h(s->o)) && (d->k_rel = up_h(s->k_rel)) &&
             (d->pw1 = up_h(s->pw1)) && (d->pw2 = up_h(s->pw2)) != NULL;
    }
    if (!ok) {
        fprintf(stderr, "media-cuda: audio weight upload failed, using the host conformer\n");
        free(ac->al); free(ac);                    // partial uploads leaked (startup OOM only)
        return NULL;
    }
    return ac;
}

static int ensure_abufs(struct acuda *ac, int T) {
    if (T <= ac->t_cap) return 0;
    int ne = ac->ne, ao = ac->ao, nf = ac->n_feat, ce = ac->n_embd;
    float **bufs[] = { &ac->F, &ac->X, &ac->H, &ac->G, &ac->D, &ac->Q, &ac->K, &ac->V,
                       &ac->P, &ac->Y, &ac->rows };
    size_t sz[] = { (size_t)T * nf, (size_t)T * ne, (size_t)T * ne, (size_t)T * 4 * ne,
                    (size_t)T * 2 * ne, (size_t)T * ne, (size_t)T * ne, (size_t)T * ne,
                    (size_t)13 * ne, (size_t)T * ao, (size_t)T * ce };
    for (int i = 0; i < 11; i++) {
        cudaFree(*bufs[i]);
        if (cudaMalloc(bufs[i], sz[i] * 4) != cudaSuccess) { ac->t_cap = 0; return -1; }
    }
    ac->t_cap = T;
    return 0;
}

// ---- the conformer ----------------------------------------------------------

extern "C" float *a_blocks_cuda(struct media *md, const float *F, int T, int n_feat, int *n_tokens) {
    if (md->audio_cuda == (void *)-1) return NULL;     // init already failed once
    struct acuda *ac = (struct acuda *)md->audio_cuda;
    if (!ac) {
        ac = acuda_init(md);
        md->audio_cuda = ac ? (void *)ac : (void *)-1;
        if (!ac) return NULL;
    }
    const int ne = ac->ne, nh = ac->nh, dh = ac->dh, ao = ac->ao, n_embd = ac->n_embd;
    const float eps = 1e-6f;
    ac->n_feat = n_feat;
    if (ensure_abufs(ac, T) != 0) return NULL;
    cudaMemcpy(ac->F, F, (size_t)T * n_feat * 4, cudaMemcpyHostToDevice);

    dim3 blk(TS, TS);
    #define GG(rows, cols) dim3(((cols) + BT - 1) / BT, ((rows) + BT - 1) / BT)
    const size_t Tne = (size_t)T * ne;
    const int e_ne  = (int)((Tne + 255) / 256);                    // elementwise grid, [T][ne]
    const int e_4ne = (int)(((size_t)T * 4 * ne + 255) / 256);
    const float q_scale = (1.0f / sqrtf((float)dh)) / logf(2.0f);  // same constants as the host
    const float k_scale = logf(1.0f + expf(1.0f)) / logf(2.0f);

    k_gemm<<<GG(T, ne), blk>>>(ac->X, ac->F, ac->inp_proj, T, n_feat, ne);   // project F in

    for (int L = 0; L < ac->n_layer; L++) {
        const struct ald *a = &ac->al[L];

        // FFN 1 (half-step residual)
        k_rms<<<T, 256>>>(ac->H, ac->X, a->ffn_norm, ne, eps);
        k_gemm<<<GG(T, 4 * ne), blk>>>(ac->G, ac->H, a->ffn_up, T, ne, 4 * ne);
        k_silu<<<e_4ne, 256>>>(ac->G, (size_t)T * 4 * ne);
        k_gemm<<<GG(T, ne), blk>>>(ac->D, ac->G, a->ffn_down, T, 4 * ne, ne);
        k_rms<<<T, 256>>>(ac->D, ac->D, a->ffn_post, ne, eps);
        k_madd<<<e_ne, 256>>>(ac->X, ac->D, 0.5f, Tne);

        // chunked local attention with relative positions
        k_rms<<<T, 256>>>(ac->H, ac->X, a->attn_pre, ne, eps);
        k_gemm<<<GG(T, ne), blk>>>(ac->Q, ac->H, a->q, T, ne, ne);
        k_gemm<<<GG(T, ne), blk>>>(ac->K, ac->H, a->k, T, ne, ne);
        k_gemm<<<GG(T, ne), blk>>>(ac->V, ac->H, a->v, T, ne, ne);
        k_gemm<<<GG(13, ne), blk>>>(ac->P, ac->rpe, a->k_rel, 13, ne, ne);
        k_qk_scale<<<e_ne, 256>>>(ac->Q, ac->K, a->pds, dh, q_scale, k_scale, Tne);
        k_local_attn<<<dim3(T, nh), dh>>>(ac->D, ac->Q, ac->K, ac->V, ac->P, ne, dh);
        k_gemm<<<GG(T, ne), blk>>>(ac->H, ac->D, a->o, T, ne, ne);
        k_rms<<<T, 256>>>(ac->H, ac->H, a->attn_post, ne, eps);
        k_add<<<e_ne, 256>>>(ac->X, ac->H, Tne);

        // convolution module
        k_rms<<<T, 256>>>(ac->H, ac->X, a->norm_conv, ne, eps);
        k_gemm<<<GG(T, 2 * ne), blk>>>(ac->G, ac->H, a->pw1, T, ne, 2 * ne);
        k_glu<<<e_ne, 256>>>(ac->D, ac->G, ne, Tne);
        k_depthwise<<<e_ne, 256>>>(ac->H, ac->D, a->dw, ne, Tne);
        k_rms<<<T, 256>>>(ac->H, ac->H, a->conv_norm, ne, eps);
        k_silu<<<e_ne, 256>>>(ac->H, Tne);
        k_gemm<<<GG(T, ne), blk>>>(ac->D, ac->H, a->pw2, T, ne, ne);
        k_add<<<e_ne, 256>>>(ac->X, ac->D, Tne);

        // FFN 2 (half-step residual)
        k_rms<<<T, 256>>>(ac->H, ac->X, a->ffn_norm1, ne, eps);
        k_gemm<<<GG(T, 4 * ne), blk>>>(ac->G, ac->H, a->ffn_up1, T, ne, 4 * ne);
        k_silu<<<e_4ne, 256>>>(ac->G, (size_t)T * 4 * ne);
        k_gemm<<<GG(T, ne), blk>>>(ac->D, ac->G, a->ffn_down1, T, 4 * ne, ne);
        k_rms<<<T, 256>>>(ac->D, ac->D, a->ffn_post1, ne, eps);
        k_madd<<<e_ne, 256>>>(ac->X, ac->D, 0.5f, Tne);

        // layer output norm
        k_rms<<<T, 256>>>(ac->X, ac->X, a->ln2, ne, eps);
    }

    // out proj -> + bias -> plain RMS -> the LLM-width projection
    k_gemm<<<GG(T, ao), blk>>>(ac->Y, ac->X, ac->out_proj, T, ne, ao);
    k_addbias<<<(int)(((size_t)T * ao + 255) / 256), 256>>>(ac->Y, ac->out_proj_b, ao, (size_t)T * ao);
    k_rms<<<T, 256>>>(ac->Y, ac->Y, NULL, ao, eps);
    k_gemm<<<GG(T, n_embd), blk>>>(ac->rows, ac->Y, ac->mm_a, T, ao, n_embd);
    #undef GG

    float *out = (float *)malloc((size_t)T * n_embd * 4);
    if (!out) return NULL;
    VC_CHECK(cudaMemcpy(out, ac->rows, (size_t)T * n_embd * 4, cudaMemcpyDeviceToHost));
    VC_CHECK(cudaGetLastError());
    *n_tokens = T;
    return out;
}

extern "C" void a_cuda_free(struct media *md) {
    struct acuda *ac = (struct acuda *)md->audio_cuda;
    if (!ac || md->audio_cuda == (void *)-1) return;
    for (int L = 0; L < ac->n_layer; L++) {
        struct ald *d = &ac->al[L];
        cudaFree(d->ffn_norm); cudaFree(d->ffn_post); cudaFree(d->ffn_norm1); cudaFree(d->ffn_post1);
        cudaFree(d->attn_pre); cudaFree(d->attn_post); cudaFree(d->pds);
        cudaFree(d->norm_conv); cudaFree(d->conv_norm); cudaFree(d->ln2); cudaFree(d->dw);
        cudaFree(d->ffn_up); cudaFree(d->ffn_down); cudaFree(d->ffn_up1); cudaFree(d->ffn_down1);
        cudaFree(d->q); cudaFree(d->k); cudaFree(d->v); cudaFree(d->o); cudaFree(d->k_rel);
        cudaFree(d->pw1); cudaFree(d->pw2);
    }
    cudaFree(ac->inp_proj); cudaFree(ac->out_proj); cudaFree(ac->mm_a);
    cudaFree(ac->out_proj_b); cudaFree(ac->rpe);
    cudaFree(ac->F); cudaFree(ac->X); cudaFree(ac->H); cudaFree(ac->G); cudaFree(ac->D);
    cudaFree(ac->Q); cudaFree(ac->K); cudaFree(ac->V); cudaFree(ac->P); cudaFree(ac->Y); cudaFree(ac->rows);
    free(ac->al);
    free(ac);
    md->audio_cuda = NULL;
}
