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

// C[t][n] = A[t][k] . W[n][k] — classic 16x16 shared-memory tiles, weights f16.
#define TS 16
static __global__ void k_gemm(float *C, const float *A, const __half *W, int T, int K, int N) {
    __shared__ float a[TS][TS], w[TS][TS];
    int tn = blockIdx.x * TS + threadIdx.x;             // output column = weight row
    int tt = blockIdx.y * TS + threadIdx.y;             // position
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TS) {
        int wr = blockIdx.x * TS + threadIdx.y;         // weight row this thread stages
        a[threadIdx.y][threadIdx.x] = tt < T ? A[(size_t)tt * K + k0 + threadIdx.x] : 0.0f;
        w[threadIdx.y][threadIdx.x] = wr < N ? __half2float(W[(size_t)wr * K + k0 + threadIdx.x]) : 0.0f;
        __syncthreads();
        for (int i = 0; i < TS; i++) acc += a[threadIdx.y][i] * w[threadIdx.x][i];
        __syncthreads();
    }
    if (tt < T && tn < N) C[(size_t)tt * N + tn] = acc;
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
    dim3 g_sq((ne + TS - 1) / TS, (np + TS - 1) / TS);          // [np][768] outputs
    dim3 g_up((4 * ne + TS - 1) / TS, (np + TS - 1) / TS);      // [np][3072]
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
    k_gemm<<<dim3((md->n_embd + TS - 1) / TS, (n + TS - 1) / TS), blk>>>
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
