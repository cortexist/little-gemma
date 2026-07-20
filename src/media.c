// Multimodal embedders for the gemma-4 mmproj files (see media.h).
// Raw data in, embedding rows out — file decoding lives in mmcat (little-gemma-tools).
// Everything runs on the host in f32: even the 16-block legacy vision
// transformer is a few hundred GFLOP per image, done once per prompt.
// The math mirrors llama.cpp's clip graphs (gemma4uv/gemma4ua for the 12B,
// gemma4v for E2B/E4B vision) so outputs can be compared against that
// implementation as an oracle — which is how the 12B and gemma4a ports were
// verified to 4-5 decimals, stage by stage.
//
// E2B/E4B AUDIO (gemma4a, a 12-block conformer) was implemented, verified
// against the reference, dropped when every pipeline confabulated on it, and
// REVIVED behind LG_GEMMA4A=1 when the QAT releases shipped a fixed audio
// encoder (2026-07-19: the confabulation isolated to the ORIGINAL mmproj's
// encoder export — the QAT mmproj transcribes verbatim, even under the old
// backbone). Default stays the Whisper lane: the conformer is speech-only,
// and the original mmprojs still circulate.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

#include "gguf.h"
#include "quant.h"
#include "media.h"
#include "media-internal.h"      // struct media/vlayer + the GPU encoder hooks

// ---- small math (f32 host) --------------------------------------------------

// pytorch LayerNorm, in place: (x - mean) / sqrt(var + eps) * w + b.
static void layernorm(float *x, const float *w, const float *b, int n, float eps) {
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= (float)n;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    var /= (float)n;
    float s = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < n; i++) x[i] = (x[i] - mean) * s * w[i] + b[i];
}

// Plain RMS norm (no weight), in place.
static void rmsnorm_plain(float *x, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float s = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) x[i] *= s;
}

// RMS norm with a weight, in place over one row.
static void rmsnorm_w(float *x, const float *w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float s = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) x[i] *= s * w[i];
}

static float gelu_quick(float x) { return x / (1.0f + expf(-1.702f * x)); }
static float silu(float x) { return x / (1.0f + expf(-x)); }

// Dot product, written naively ON PURPOSE: this file is compiled with
// fast-math (see CMakeLists), which licenses the compiler to reassociate the
// sum into a SIMD reduction. A hand-unrolled multi-accumulator version was
// tried and was WORSE — it defeats MSVC's vectorizer pattern match.
static float dotf(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int j = 0; j < n; j++) s += a[j] * b[j];
    return s;
}

// out[m] = W . x, unpacking each (f16/f32) weight row on the fly — the same
// shape as the CPU backend's matmul_q, OpenMP across the independent rows.
static void matvec(float *out, const struct gguf_tensor *t, const float *x, int k, int m) {
    const int blck = ggml_blck_size(t->type);
    const size_t row_bytes = (size_t)(k / blck) * ggml_type_size(t->type);
    const unsigned char *base = t->data;
    #pragma omp parallel
    {
        float *buf = malloc((size_t)k * sizeof(float));
        int i;
        #pragma omp for schedule(static)
        for (i = 0; i < m; i++) {
            dequantize_into(t->type, base + (size_t)i * row_bytes, buf, k);
            out[i] = dotf(buf, x, k);
        }
        free(buf);
    }
}

// out[t][r] = W . x_t for all T rows at once — weight rows are unpacked ONCE,
// a block at a time, and every position is dotted against the block while it
// is cache-hot. X is [T][k] contiguous, out is [T][m]. The block matters:
// one-row-at-a-time re-streams all of X from memory per weight row (~90 GB of
// traffic per gemma4v layer at 266 tokens), and the encoder goes memory-bound
// — blocking by 64 cuts that traffic 64-fold.
#define MM_RB 64
static void matmat(float *out, const struct gguf_tensor *t, const float *X, int k, int m, int T) {
    const int blck = ggml_blck_size(t->type);
    const size_t row_bytes = (size_t)(k / blck) * ggml_type_size(t->type);
    const unsigned char *base = t->data;
    #pragma omp parallel
    {
        float *buf = malloc((size_t)MM_RB * k * sizeof(float));
        int rb;
        #pragma omp for schedule(static)
        for (rb = 0; rb < (m + MM_RB - 1) / MM_RB; rb++) {
            int r0 = rb * MM_RB, rn = m - r0 < MM_RB ? m - r0 : MM_RB;
            for (int r = 0; r < rn; r++)
                dequantize_into(t->type, base + (size_t)(r0 + r) * row_bytes, buf + (size_t)r * k, k);
            for (int tt = 0; tt < T; tt++) {
                const float *x = X + (size_t)tt * k;
                float *o = out + (size_t)tt * m + r0;
                for (int r = 0; r < rn; r++) o[r] = dotf(buf + (size_t)r * k, x, k);
            }
        }
        free(buf);
    }
}

// ---- open / free ------------------------------------------------------------

static const float *fptr(const struct gguf_context *ctx, const char *name) {
    const struct gguf_tensor *t = gguf_find_tensor(ctx, name);
    return t ? (const float *)t->data : NULL;
}
static const float *vfptr(struct media *md, const char *fmt, int L) {
    char nm[64];
    snprintf(nm, sizeof nm, fmt, L);
    return fptr(md->ctx, nm);
}
static const struct gguf_tensor *vtens(struct media *md, const char *fmt, int L) {
    char nm[64];
    snprintf(nm, sizeof nm, fmt, L);
    return gguf_find_tensor(md->ctx, nm);
}

static int legacy_audio_open(struct media *md);   // gemma4a, defined below

static int legacy_vision_open(struct media *md) {
    struct gguf_context *ctx = md->ctx;
    md->v_embd  = (int)gguf_get_u32(ctx, "clip.vision.embedding_length", 0);
    md->v_head  = (int)gguf_get_u32(ctx, "clip.vision.attention.head_count", 0);
    md->v_layer = (int)gguf_get_u32(ctx, "clip.vision.block_count", 0);
    md->patch   = (int)gguf_get_u32(ctx, "clip.vision.patch_size", 0);
    md->v_merge = 3;                           // pooling kernel (reference hardcodes it)
    md->v_patch16 = gguf_find_tensor(ctx, "v.patch_embd.weight");
    md->mm_v = gguf_find_tensor(ctx, "mm.input_projection.weight");
    const struct gguf_tensor *pos = gguf_find_tensor(ctx, "v.position_embd.weight");
    if (!md->v_embd || !md->v_head || !md->v_layer || md->patch != 16 ||
        !md->v_patch16 || !md->mm_v || !pos || (int)md->mm_v->dims[1] != md->n_embd) {
        fprintf(stderr, "media: gemma4v tensors missing or mismatched\n");
        return -1;
    }
    md->v_pos = (const float *)pos->data;
    md->pos_size = (int)pos->dims[1];
    md->vl = calloc((size_t)md->v_layer, sizeof *md->vl);
    if (!md->vl) return -1;
    for (int L = 0; L < md->v_layer; L++) {
        struct vlayer *v = &md->vl[L];
        v->ln1       = vfptr(md, "v.blk.%d.ln1.weight", L);
        v->ln2       = vfptr(md, "v.blk.%d.ln2.weight", L);
        v->attn_post = vfptr(md, "v.blk.%d.attn_post_norm.weight", L);
        v->ffn_post  = vfptr(md, "v.blk.%d.ffn_post_norm.weight", L);
        v->qn        = vfptr(md, "v.blk.%d.attn_q_norm.weight", L);
        v->kn        = vfptr(md, "v.blk.%d.attn_k_norm.weight", L);
        v->q    = vtens(md, "v.blk.%d.attn_q.weight", L);
        v->k    = vtens(md, "v.blk.%d.attn_k.weight", L);
        v->v    = vtens(md, "v.blk.%d.attn_v.weight", L);
        v->o    = vtens(md, "v.blk.%d.attn_out.weight", L);
        v->gate = vtens(md, "v.blk.%d.ffn_gate.weight", L);
        v->up   = vtens(md, "v.blk.%d.ffn_up.weight", L);
        v->down = vtens(md, "v.blk.%d.ffn_down.weight", L);
        if (!v->ln1 || !v->ln2 || !v->attn_post || !v->ffn_post || !v->qn || !v->kn ||
            !v->q || !v->k || !v->v || !v->o || !v->gate || !v->up || !v->down) {
            fprintf(stderr, "media: gemma4v block %d tensors missing\n", L);
            return -1;
        }
    }
    return 0;
}

struct media *media_open(const char *path, int n_embd) {
    struct gguf_context *ctx = load_gguf(path);
    if (!ctx) return NULL;

    const char *vp = gguf_get_str(ctx, "clip.vision.projector_type", "");
    const char *ap = gguf_get_str(ctx, "clip.audio.projector_type", "");
    int unified  = strcmp(vp, "gemma4uv") == 0 && strcmp(ap, "gemma4ua") == 0;
    int legacy_v = strcmp(vp, "gemma4v") == 0;
    if (!unified && !legacy_v) {
        fprintf(stderr, "media: %s is a '%s'/'%s' mmproj; supported are the "
                        "encoder-free gemma4uv/gemma4ua (12B) and, vision only, "
                        "the legacy gemma4v transformer (E2B/E4B)\n", path, vp, ap);
        free_gguf(ctx);
        return NULL;
    }

    struct media *md = calloc(1, sizeof *md);
    if (!md) { free_gguf(ctx); return NULL; }
    md->ctx = ctx;
    md->n_embd = n_embd;

    if (legacy_v) {
        // E2B/E4B carry two legacy towers in one mmproj: the 16-block vision
        // transformer (gemma4v) and the 12-block audio conformer (gemma4a).
        // Vision always opens; audio only under LG_GEMMA4A=1 (see the header
        // note) — without it, audio frames error with a pointer to the switch.
        md->legacy_v = 1;
        md->frame = 640;                       // the wire stays raw 16 kHz PCM
        if (legacy_vision_open(md) != 0) { media_free(md); return NULL; }
        if (strcmp(ap, "gemma4a") == 0 && getenv("LG_GEMMA4A")) {
            md->legacy_a = 1;
            if (legacy_audio_open(md) != 0) { media_free(md); return NULL; }
        }
        return md;
    }

    md->v_patch_w = gguf_find_tensor(ctx, "v.patch_embd.weight");
    md->v_patch_b = fptr(ctx, "v.patch_embd.bias");
    md->vn1w = fptr(ctx, "v.patch_norm.1.weight"); md->vn1b = fptr(ctx, "v.patch_norm.1.bias");
    md->vn2w = fptr(ctx, "v.patch_norm.2.weight"); md->vn2b = fptr(ctx, "v.patch_norm.2.bias");
    md->vn3w = fptr(ctx, "v.patch_norm.3.weight"); md->vn3b = fptr(ctx, "v.patch_norm.3.bias");
    md->mm_v = gguf_find_tensor(ctx, "mm.input_projection.weight");
    md->mm_a = gguf_find_tensor(ctx, "mm.a.input_projection.weight");
    const struct gguf_tensor *pos = gguf_find_tensor(ctx, "v.position_embd.weight");
    if (!md->v_patch_w || !md->v_patch_b || !md->vn1w || !md->vn1b || !md->vn2w || !md->vn2b ||
        !md->vn3w || !md->vn3b || !md->mm_v || !md->mm_a || !pos) {
        fprintf(stderr, "media: %s is missing gemma4uv/gemma4ua tensors\n", path);
        media_free(md);
        return NULL;
    }
    md->v_pos = (const float *)pos->data;
    md->pos_size = (int)pos->dims[1];

    // geometry derived from the tensors themselves: the patch side from the
    // embed's input width (pixels * 3 channels), the audio frame likewise
    int patch_in = (int)md->v_patch_w->dims[0];
    md->patch = (int)lroundf(sqrtf((float)patch_in / 3.0f));
    md->frame = (int)md->mm_a->dims[0];
    if (md->patch * md->patch * 3 != patch_in ||
        (int)md->mm_v->dims[1] != n_embd || (int)md->mm_a->dims[1] != n_embd) {
        fprintf(stderr, "media: %s geometry mismatch (patch_in %d, out %d/%d vs n_embd %d)\n",
                path, patch_in, (int)md->mm_v->dims[1], (int)md->mm_a->dims[1], n_embd);
        media_free(md);
        return NULL;
    }
    return md;
}

void media_free(struct media *md) {
    if (!md) return;
    v_gpu_free(md);   // no-op in the CPU build (stub); frees GPU state in GPU builds
    free_gguf(md->ctx);
    free(md->vl);
    free(md->al); free(md->mel_filt); free(md->rpe);
    free(md);
}

int media_patch(const struct media *md) { return md->legacy_v ? md->patch * md->v_merge : md->patch; }
int media_frame(const struct media *md) { return md->frame; }

// ---- vision, unified (gemma4uv) ----------------------------------------------

static float *uv_embed_image(struct media *md, const uint8_t *rgb, int w, int h, int *n_tokens) {
    const int P = md->patch, pin = P * P * 3, ne = md->n_embd;
    int n_cols = w / P, n_rows = h / P, n = n_cols * n_rows;
    if (n_cols > md->pos_size || n_rows > md->pos_size) {
        fprintf(stderr, "media: image %dx%d exceeds the %d-patch position table\n", w, h, md->pos_size);
        return NULL;
    }

    // batched on purpose: the two linears carry ~80 MB of f16 weights, and a
    // patch-at-a-time matvec re-unpacks and re-streams all of it per patch
    // (266 x 80 MB for a typical image) — matmat unpacks each row once
    float *out = malloc((size_t)n * ne * sizeof(float));
    float *E   = malloc((size_t)n * ne * sizeof(float));
    float *F   = malloc((size_t)n * pin * sizeof(float));
    if (!out || !E || !F) { free(out); free(E); free(F); return NULL; }

    int p;
    #pragma omp parallel for schedule(static)
    for (p = 0; p < n; p++) {
        int col = p % n_cols, row = p / n_cols;
        float *vec = F + (size_t)p * pin;
        // flatten the patch the way im2col over a planar [W,H,3] image does:
        // channel-major, row-major pixels within the channel, in [0,1]
        for (int c = 0; c < 3; c++)
            for (int ky = 0; ky < P; ky++)
                for (int kx = 0; kx < P; kx++)
                    vec[(c * P + ky) * P + kx] =
                        (float)rgb[3 * ((size_t)(row * P + ky) * w + (col * P + kx)) + c] / 255.0f;
        layernorm(vec, md->vn1w, md->vn1b, pin, 1e-5f);
    }
    matmat(E, md->v_patch_w, F, pin, ne, n);
    #pragma omp parallel for schedule(static)
    for (p = 0; p < n; p++) {
        float *e = E + (size_t)p * ne;
        int col = p % n_cols, row = p / n_cols;
        const float *ex = md->v_pos + (size_t)col * ne;                       // x table
        const float *ey = md->v_pos + ((size_t)md->pos_size + row) * ne;      // y table
        for (int i = 0; i < ne; i++) e[i] += md->v_patch_b[i];
        layernorm(e, md->vn2w, md->vn2b, ne, 1e-5f);
        for (int i = 0; i < ne; i++) e[i] += ex[i] + ey[i];
        layernorm(e, md->vn3w, md->vn3b, ne, 1e-5f);
        rmsnorm_plain(e, ne, 1e-6f);
    }
    matmat(out, md->mm_v, E, ne, ne, n);
    free(F); free(E);
    *n_tokens = n;
    return out;
}

// ---- vision, legacy (gemma4v: E2B/E4B 16-block ViT) ---------------------------
// Mirrors clip_graph_gemma4v + build_vit: RMS norms throughout, no biases,
// per-head q/k norms, 2D NeoX RoPE (first half of each head by patch column,
// second half by row, theta 100), a plain RMS norm on V, full attention at
// scale 1.0, gelu_quick-gated FFNs, then a 3x3 average pool, x sqrt(768),
// the [768 -> n_embd] projection, and a PLAIN rms norm AFTER it (the
// unified path norms BEFORE its projection — they differ on purpose).

static void rope2d(float *X, int np, int ne, int nh, int n_cols, float theta) {
    const int dh = ne / nh, half = dh / 2, quarter = half / 2;
    int p;
    #pragma omp parallel for schedule(static)
    for (p = 0; p < np; p++) {
        int posv[2] = { p % n_cols, p / n_cols };           // col for half 0, row for half 1
        for (int h = 0; h < nh; h++) {
            float *d = X + (size_t)p * ne + h * dh;
            for (int sec = 0; sec < 2; sec++) {             // the two 32-dim halves
                float *s = d + sec * half;
                for (int i = 0; i < quarter; i++) {         // NeoX pairs (i, i+quarter)
                    float freq = powf(theta, -2.0f * (float)i / (float)half);
                    float ang = (float)posv[sec] * freq;
                    float c = cosf(ang), sn = sinf(ang);
                    float a = s[i], b = s[i + quarter];
                    s[i]           = a * c - b * sn;
                    s[i + quarter] = a * sn + b * c;
                }
            }
        }
    }
}

static float *v_embed_image(struct media *md, const uint8_t *rgb, int w, int h, int *n_tokens) {
    const int P = md->patch, ne = md->v_embd, nh = md->v_head, dh = ne / nh;
    const int mg = md->v_merge;
    const float eps = 1e-6f;
    int n_cols = w / P, n_rows = h / P, np = n_cols * n_rows;
    if (n_cols > md->pos_size || n_rows > md->pos_size) {
        fprintf(stderr, "media: image %dx%d exceeds the %d-entry position table\n", w, h, md->pos_size);
        return NULL;
    }

    // patch embed: conv 16x16 stride 16, no bias, input scaled to [-1, 1]
    float *F = malloc((size_t)np * P * P * 3 * sizeof(float));
    float *X = malloc((size_t)np * ne * sizeof(float));
    if (!F || !X) { free(F); free(X); return NULL; }
    for (int p = 0; p < np; p++) {
        int col = p % n_cols, row = p / n_cols;
        float *vec = F + (size_t)p * P * P * 3;
        for (int c = 0; c < 3; c++)
            for (int ky = 0; ky < P; ky++)
                for (int kx = 0; kx < P; kx++)
                    vec[(c * P + ky) * P + kx] =
                        (float)rgb[3 * ((size_t)(row * P + ky) * w + (col * P + kx)) + c] / 255.0f * 2.0f - 1.0f;
    }
    matmat(X, md->v_patch16, F, P * P * 3, ne, np);
    free(F);
    for (int p = 0; p < np; p++) {                          // learned 2-axis positions
        int col = p % n_cols, row = p / n_cols;
        const float *ex = md->v_pos + (size_t)col * ne;
        const float *ey = md->v_pos + ((size_t)md->pos_size + row) * ne;
        for (int i = 0; i < ne; i++) X[(size_t)p * ne + i] += ex[i] + ey[i];
    }

    float *H = malloc((size_t)np * ne * sizeof(float));
    float *Q = malloc((size_t)np * ne * sizeof(float));
    float *K = malloc((size_t)np * ne * sizeof(float));
    float *V = malloc((size_t)np * ne * sizeof(float));
    float *D = malloc((size_t)np * ne * sizeof(float));
    float *G = malloc((size_t)np * 4 * ne * sizeof(float));
    float *U = malloc((size_t)np * 4 * ne * sizeof(float));
    if (!H || !Q || !K || !V || !D || !G || !U) goto fail;

    // every per-position loop below is OpenMP'd (index declared outside the
    // for: MSVC's C-mode OpenMP insists) — at 16 layers even the elementwise
    // passes add up, the gelu alone is a hundred million expf calls
    int p;
    for (int L = 0; L < md->v_layer; L++) {
        const struct vlayer *v = &md->vl[L];

        // ---- attention ----
        #pragma omp parallel for schedule(static)
        for (p = 0; p < np; p++) {
            memcpy(H + (size_t)p * ne, X + (size_t)p * ne, (size_t)ne * 4);
            rmsnorm_w(H + (size_t)p * ne, v->ln1, ne, eps);
        }
        matmat(Q, v->q, H, ne, ne, np);
        matmat(K, v->k, H, ne, ne, np);
        matmat(V, v->v, H, ne, ne, np);
        #pragma omp parallel for schedule(static)
        for (p = 0; p < np; p++)
            for (int hh = 0; hh < nh; hh++) {
                rmsnorm_w(Q + (size_t)p * ne + hh * dh, v->qn, dh, eps);
                rmsnorm_w(K + (size_t)p * ne + hh * dh, v->kn, dh, eps);
                rmsnorm_plain(V + (size_t)p * ne + hh * dh, dh, eps);
            }
        rope2d(Q, np, ne, nh, n_cols, 100.0f);
        rope2d(K, np, ne, nh, n_cols, 100.0f);
        // full attention, scale 1.0, in query blocks: one query at a time
        // would re-stream all of K and V per query (14 MB x 2394 queries, per
        // layer) — per BLOCK, each key/value row is loaded once and used 64x
        #pragma omp parallel
        {
            enum { QB = 64 };
            // local copies: OpenMP outlines this block into its own function
            // and rewrites captured variables as loads through a shared-frame
            // pointer — MSVC's vectorizer then refuses every loop bounded by
            // them ("upper bound is not loop-invariant", C5002 reason 501)
            const int NP = np, NH = nh, DH = dh, NE = ne;
            const float *Kl = K, *Vl = V, *Ql = Q;
            float *Dl = D;
            float *att = malloc((size_t)QB * NP * sizeof(float));
            int qb;
            #pragma omp for schedule(static)
            for (qb = 0; qb < (NP + QB - 1) / QB; qb++) {
                int q0 = qb * QB, qn = NP - q0 < QB ? NP - q0 : QB;
                for (int hh = 0; hh < NH; hh++) {
                    for (int kk = 0; kk < NP; kk++) {
                        const float *kh = Kl + (size_t)kk * NE + hh * DH;
                        for (int q = 0; q < qn; q++)
                            att[(size_t)q * NP + kk] = dotf(Ql + (size_t)(q0 + q) * NE + hh * DH, kh, DH);
                    }
                    for (int q = 0; q < qn; q++) {          // softmax per query row
                        float *a = att + (size_t)q * NP;
                        float mx = -1e30f;
                        for (int kk = 0; kk < NP; kk++) if (a[kk] > mx) mx = a[kk];
                        // map and reduce in separate loops: fused, the expf
                        // call blocks vectorizing the sum and vice versa
                        for (int kk = 0; kk < NP; kk++) a[kk] = expf(a[kk] - mx);
                        float sum = 0.0f;
                        for (int kk = 0; kk < NP; kk++) sum += a[kk];
                        float inv = 1.0f / sum;
                        for (int kk = 0; kk < NP; kk++) a[kk] *= inv;
                        float *outr = Dl + (size_t)(q0 + q) * NE + hh * DH;
                        for (int i = 0; i < DH; i++) outr[i] = 0.0f;
                    }
                    for (int kk = 0; kk < NP; kk++) {       // weighted V, values outer
                        const float *vh = Vl + (size_t)kk * NE + hh * DH;
                        for (int q = 0; q < qn; q++) {
                            float *outr = Dl + (size_t)(q0 + q) * NE + hh * DH;
                            float wgt = att[(size_t)q * NP + kk];
                            for (int i = 0; i < DH; i++) outr[i] += wgt * vh[i];
                        }
                    }
                }
            }
            free(att);
        }
        matmat(H, v->o, D, ne, ne, np);
        #pragma omp parallel for schedule(static)
        for (p = 0; p < np; p++) {
            rmsnorm_w(H + (size_t)p * ne, v->attn_post, ne, eps);
            for (int i = 0; i < ne; i++) X[(size_t)p * ne + i] += H[(size_t)p * ne + i];
        }

        // ---- gated FFN: gelu_quick(gate) * up -> down ----
        #pragma omp parallel for schedule(static)
        for (p = 0; p < np; p++) {
            memcpy(H + (size_t)p * ne, X + (size_t)p * ne, (size_t)ne * 4);
            rmsnorm_w(H + (size_t)p * ne, v->ln2, ne, eps);
        }
        matmat(G, v->gate, H, ne, 4 * ne, np);
        matmat(U, v->up, H, ne, 4 * ne, np);
        #pragma omp parallel for schedule(static)
        for (p = 0; p < np; p++) {
            float *g = G + (size_t)p * 4 * ne;
            const float *u = U + (size_t)p * 4 * ne;
            for (int i = 0; i < 4 * ne; i++) g[i] = gelu_quick(g[i]) * u[i];
        }
        matmat(D, v->down, G, 4 * ne, ne, np);
        #pragma omp parallel for schedule(static)
        for (p = 0; p < np; p++) {
            rmsnorm_w(D + (size_t)p * ne, v->ffn_post, ne, eps);
            for (int i = 0; i < ne; i++) X[(size_t)p * ne + i] += D[(size_t)p * ne + i];
        }
    }

    // 3x3 average pool over the patch grid, x sqrt(768), project, then norm
    {
        int out_x = n_cols / mg, out_y = n_rows / mg, n = out_x * out_y;
        float sc = sqrtf((float)ne);
        float *pooled = malloc((size_t)n * ne * sizeof(float));
        float *rows = malloc((size_t)n * md->n_embd * sizeof(float));
        if (!pooled || !rows) { free(pooled); free(rows); goto fail; }
        for (int ty = 0; ty < out_y; ty++)
            for (int tx = 0; tx < out_x; tx++) {
                float *po = pooled + (size_t)(ty * out_x + tx) * ne;
                for (int i = 0; i < ne; i++) po[i] = 0.0f;
                for (int dy = 0; dy < mg; dy++)
                    for (int dx = 0; dx < mg; dx++) {
                        const float *xp = X + (size_t)((ty * mg + dy) * n_cols + (tx * mg + dx)) * ne;
                        for (int i = 0; i < ne; i++) po[i] += xp[i];
                    }
                for (int i = 0; i < ne; i++) po[i] *= sc / (float)(mg * mg);
            }
        matmat(rows, md->mm_v, pooled, ne, md->n_embd, n);
        for (int t = 0; t < n; t++)
            rmsnorm_plain(rows + (size_t)t * md->n_embd, md->n_embd, eps);
        free(pooled);
        free(H); free(Q); free(K); free(V); free(D); free(G); free(U); free(X);
        *n_tokens = n;
        return rows;
    }

fail:
    free(H); free(Q); free(K); free(V); free(D); free(G); free(U); free(X);
    return NULL;
}

// LG_MEDIA_VERIFY: the GPU rows against the host path's, for any embedder pair.
static void verify_rows(const struct media *md, const float *rows, const float *ref, int n, int n2) {
    double mx = 0;
    if (ref && n2 == n)
        for (size_t i = 0; i < (size_t)n2 * md->n_embd; i++) {
            double d = fabs((double)rows[i] - ref[i]);
            if (d > mx) mx = d;
        }
    fprintf(stderr, "media: gpu vs host max |diff| %.3g over %d rows\n", mx, n2);
    // Reading the number: the GPU rows are BIT-IDENTICAL across devices (verified
    // A5000 vs Orin, 2026-07-02); what varies is this HOST reference — fast-math
    // OpenMP differs per CPU arch (x86 vs ARM read ~0.01 vs ~0.15 on the unified
    // path, whose rows are unnormalized, |x| ~ 50 — that is ~3e-3 relative).
}

float *media_embed_image(struct media *md, const uint8_t *rgb, int w, int h, int *n_tokens) {
    int P = media_patch(md);
    if (w <= 0 || h <= 0 || w % P || h % P) {
        fprintf(stderr, "media: image %dx%d is not a multiple of the %d-pixel patch\n", w, h, P);
        return NULL;
    }
    // try the GPU embedder (NULL in the CPU build / when unusable), then host
    float *(*host)(struct media *, const uint8_t *, int, int, int *) =
        md->legacy_v ? v_embed_image : uv_embed_image;
    float *rows = md->legacy_v ? v_embed_image_gpu(md, rgb, w, h, n_tokens)
                               : uv_embed_image_gpu(md, rgb, w, h, n_tokens);
    if (rows && getenv("LG_MEDIA_VERIFY")) {        // host path as numeric oracle
        int n2 = 0;
        float *ref = host(md, rgb, w, h, &n2);
        verify_rows(md, rows, ref, *n_tokens, n2);
        free(ref);
    }
    if (rows) return rows;                          // NULL: no GPU / unusable -> host
    return host(md, rgb, w, h, n_tokens);
}

// ================  legacy audio: the gemma4a conformer (E2B/E4B)  ============
// A log-mel spectrogram, two stride-2 convolutions, and 12 conformer blocks
// (dual half-step SiLU FFNs, chunked local attention with sinusoidal relative
// positions, a gated causal depthwise-conv module). Mirrors llama.cpp's
// clip_graph_gemma4a; verified bit-for-bit against it in 2026-06. The host
// path here is the numeric oracle for the GPU encoder and the only path on
// the CPU build. LG_GEMMA4A=1 opens it (see the header note).

#define A_NFFT   512      // FFT size; the 20 ms (320-sample) Hann window zero-padded to it
#define A_WIN    320
#define A_HOP    160      // 10 ms
#define A_BINS   (A_NFFT / 2 + 1)

// In-place iterative radix-2 complex FFT (interleaved re,im), N a power of 2.
static void fft(float *a, int N) {
    for (int i = 1, j = 0; i < N; i++) {                  // bit-reversal permutation
        int bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = a[2*i], ti = a[2*i+1];
            a[2*i] = a[2*j]; a[2*i+1] = a[2*j+1];
            a[2*j] = tr; a[2*j+1] = ti;
        }
    }
    for (int len = 2; len <= N; len <<= 1) {
        double ang = -2.0 * 3.14159265358979323846 / len;
        for (int i = 0; i < N; i += len)
            for (int j = 0; j < len / 2; j++) {
                float wr = (float)cos(ang * j), wi = (float)sin(ang * j);
                float ur = a[2*(i+j)],         ui = a[2*(i+j)+1];
                float vr = a[2*(i+j+len/2)]   * wr - a[2*(i+j+len/2)+1] * wi;
                float vi = a[2*(i+j+len/2)]   * wi + a[2*(i+j+len/2)+1] * wr;
                a[2*(i+j)]       = ur + vr; a[2*(i+j)+1]       = ui + vi;
                a[2*(i+j+len/2)] = ur - vr; a[2*(i+j+len/2)+1] = ui - vi;
            }
    }
}

// HTK-scale triangular mel filterbank (no area normalization), 0..8 kHz.
static float *mel_filterbank(int n_mel) {
    float *filt = calloc((size_t)n_mel * A_BINS, sizeof(float));
    if (!filt) return NULL;
    double m_hi = 2595.0 * log10(1.0 + 8000.0 / 700.0);
    double hz[130 + 2];                                   // n_mel + 2 edges (n_mel <= 130)
    for (int i = 0; i < n_mel + 2; i++)
        hz[i] = 700.0 * (pow(10.0, (m_hi * i / (n_mel + 1)) / 2595.0) - 1.0);
    for (int m = 0; m < n_mel; m++) {
        double fl = hz[m], fc = hz[m + 1], fr = hz[m + 2];
        for (int k = 0; k < A_BINS; k++) {
            double f = k * 16000.0 / A_NFFT, w = 0.0;
            if (f >= fl && f <= fc)      w = (f - fl) / (fc - fl);
            else if (f > fc && f <= fr)  w = (fr - f) / (fr - fc);
            filt[(size_t)m * A_BINS + k] = (float)w;
        }
    }
    return filt;
}

// Natural-log mel spectrogram of 16 kHz samples — semicausal left padding of
// half a window, frame count matching the reference's PyTorch arithmetic.
// Returns [n_frames][n_mel].
static float *log_mel(const struct media *md, const float *x, int n, int *n_frames) {
    const int nm = md->n_mel;
    const int pt = (n + A_WIN / 2 - (A_WIN + 1)) / A_HOP + 1;     // frame count
    if (pt <= 0) return NULL;
    int n_pad = (pt - 1) * A_HOP + A_NFFT;                         // padded length
    if (n_pad < n + A_WIN / 2) n_pad = n + A_WIN / 2;
    float *pad = calloc((size_t)n_pad, sizeof(float));
    float *mel = malloc((size_t)pt * nm * sizeof(float));
    if (!pad || !mel) { free(pad); free(mel); return NULL; }
    memcpy(pad + A_WIN / 2, x, (size_t)n * sizeof(float));
    #pragma omp parallel
    {
        float *fr = malloc(2 * A_NFFT * sizeof(float));
        float mag[A_BINS];
        int i;
        #pragma omp for schedule(static)
        for (i = 0; i < pt; i++) {
            for (int j = 0; j < A_NFFT; j++) {
                float w = j < A_WIN ? 0.5f - 0.5f * cosf(2.0f * 3.14159265f * j / A_WIN) : 0.0f;
                fr[2*j] = w * pad[i * A_HOP + j];
                fr[2*j+1] = 0.0f;
            }
            fft(fr, A_NFFT);
            for (int k = 0; k < A_BINS; k++)
                mag[k] = sqrtf(fr[2*k] * fr[2*k] + fr[2*k+1] * fr[2*k+1]);
            for (int mi = 0; mi < nm; mi++) {
                double s = 0.0;
                const float *fl = md->mel_filt + (size_t)mi * A_BINS;
                for (int k = 0; k < A_BINS; k++) s += mag[k] * fl[k];
                mel[(size_t)i * nm + mi] = (float)log(s > 0.001 ? s : 0.001);
            }
        }
        free(fr);
    }
    free(pad);
    *n_frames = pt;
    return mel;
}

// One f32 scalar tensor by formatted name, or a default when absent.
static float ascalar(struct media *md, const char *base, int L, const char *suffix, float dflt) {
    char nm[96];
    int n = snprintf(nm, sizeof nm, base, L);
    snprintf(nm + n, sizeof nm - n, ".%s", suffix);
    const struct gguf_tensor *t = gguf_find_tensor(md->ctx, nm);
    return t ? *(const float *)t->data : dflt;
}
// The QAT range quad for one projection ("a.blk.%d.ffn_up" etc.).
static struct aclamp aclamp_of(struct media *md, const char *base, int L) {
    struct aclamp c;
    c.ilo = ascalar(md, base, L, "input_min",  -FLT_MAX);
    c.ihi = ascalar(md, base, L, "input_max",   FLT_MAX);
    c.olo = ascalar(md, base, L, "output_min", -FLT_MAX);
    c.ohi = ascalar(md, base, L, "output_max",  FLT_MAX);
    return c;
}

static int legacy_audio_open(struct media *md) {
    struct gguf_context *ctx = md->ctx;
    md->a_embd  = (int)gguf_get_u32(ctx, "clip.audio.embedding_length", 0);
    md->a_head  = (int)gguf_get_u32(ctx, "clip.audio.attention.head_count", 0);
    md->a_layer = (int)gguf_get_u32(ctx, "clip.audio.block_count", 0);
    md->n_mel   = (int)gguf_get_u32(ctx, "clip.audio.num_mel_bins", 0);
    md->a_conv0 = gguf_find_tensor(ctx, "a.conv1d.0.weight");
    md->a_conv1 = gguf_find_tensor(ctx, "a.conv1d.1.weight");
    md->a_conv0_n = fptr(ctx, "a.conv1d.0.norm.weight");
    md->a_conv1_n = fptr(ctx, "a.conv1d.1.norm.weight");
    md->a_inp_proj = gguf_find_tensor(ctx, "a.input_projection.weight");
    md->a_out_proj = gguf_find_tensor(ctx, "a.pre_encode.out.weight");
    md->a_out_proj_b = fptr(ctx, "a.pre_encode.out.bias");
    md->mm_a = gguf_find_tensor(ctx, "mm.a.input_projection.weight");
    if (!md->a_embd || !md->a_head || !md->a_layer || md->n_mel > 130 ||
        !md->a_conv0 || !md->a_conv1 || !md->a_conv0_n || !md->a_conv1_n ||
        !md->a_inp_proj || !md->a_out_proj || !md->a_out_proj_b || !md->mm_a ||
        (int)md->mm_a->dims[1] != md->n_embd) {
        fprintf(stderr, "media: gemma4a tensors missing or mismatched\n");
        return -1;
    }
    md->al = calloc((size_t)md->a_layer, sizeof *md->al);
    if (!md->al) return -1;
    for (int L = 0; L < md->a_layer; L++) {
        struct alayer *a = &md->al[L];
        a->ffn_norm  = vfptr(md, "a.blk.%d.ffn_norm.weight", L);
        a->ffn_post  = vfptr(md, "a.blk.%d.ffn_post_norm.weight", L);
        a->ffn_up    = vtens(md, "a.blk.%d.ffn_up.weight", L);
        a->ffn_down  = vtens(md, "a.blk.%d.ffn_down.weight", L);
        a->ffn_norm1 = vfptr(md, "a.blk.%d.ffn_norm_1.weight", L);
        a->ffn_post1 = vfptr(md, "a.blk.%d.ffn_post_norm_1.weight", L);
        a->ffn_up1   = vtens(md, "a.blk.%d.ffn_up_1.weight", L);
        a->ffn_down1 = vtens(md, "a.blk.%d.ffn_down_1.weight", L);
        a->attn_pre  = vfptr(md, "a.blk.%d.attn_pre_norm.weight", L);
        a->attn_post = vfptr(md, "a.blk.%d.attn_post_norm.weight", L);
        a->pds       = vfptr(md, "a.blk.%d.per_dim_scale.weight", L);
        a->q = vtens(md, "a.blk.%d.attn_q.weight", L);
        a->k = vtens(md, "a.blk.%d.attn_k.weight", L);
        a->v = vtens(md, "a.blk.%d.attn_v.weight", L);
        a->o = vtens(md, "a.blk.%d.attn_out.weight", L);
        a->k_rel = vtens(md, "a.blk.%d.attn_k_rel.weight", L);
        // the two conv-module norms are CROSS-WIRED in the file: the pre-conv
        // norm is stored as "conv_norm", the post-depthwise norm as "norm_conv"
        // (the reference loader swaps them exactly like this)
        a->norm_conv = vfptr(md, "a.blk.%d.conv_norm.weight", L);
        a->conv_norm = vfptr(md, "a.blk.%d.norm_conv.weight", L);
        a->ln2       = vfptr(md, "a.blk.%d.ln2.weight", L);
        a->dw        = vfptr(md, "a.blk.%d.conv_dw.weight", L);
        a->pw1 = vtens(md, "a.blk.%d.conv_pw1.weight", L);
        a->pw2 = vtens(md, "a.blk.%d.conv_pw2.weight", L);
        if (!a->ffn_norm || !a->ffn_post || !a->ffn_up || !a->ffn_down ||
            !a->ffn_norm1 || !a->ffn_post1 || !a->ffn_up1 || !a->ffn_down1 ||
            !a->attn_pre || !a->attn_post || !a->pds || !a->q || !a->k || !a->v ||
            !a->o || !a->k_rel || !a->norm_conv || !a->conv_norm || !a->ln2 ||
            !a->dw || !a->pw1 || !a->pw2) {
            fprintf(stderr, "media: gemma4a block %d tensors missing\n", L);
            return -1;
        }
        a->c_ffn_up   = aclamp_of(md, "a.blk.%d.ffn_up", L);
        a->c_ffn_down = aclamp_of(md, "a.blk.%d.ffn_down", L);
        a->c_ffn_up1  = aclamp_of(md, "a.blk.%d.ffn_up_1", L);
        a->c_ffn_down1 = aclamp_of(md, "a.blk.%d.ffn_down_1", L);
        a->c_q   = aclamp_of(md, "a.blk.%d.attn_q", L);
        a->c_k   = aclamp_of(md, "a.blk.%d.attn_k", L);
        a->c_v   = aclamp_of(md, "a.blk.%d.attn_v", L);
        a->c_o   = aclamp_of(md, "a.blk.%d.attn_out", L);
        a->c_pw1 = aclamp_of(md, "a.blk.%d.conv_pw1", L);
        a->c_pw2 = aclamp_of(md, "a.blk.%d.conv_pw2", L);
    }
    md->mel_filt = mel_filterbank(md->n_mel);
    // Sinusoidal relative positions, table index p = distance 12-p (so index 12
    // is "self"): emb[i] = sin(pos * 10000^(-i/(half-1))), cos in the upper half.
    int half = md->a_embd / 2;
    md->rpe = malloc((size_t)13 * md->a_embd * sizeof(float));
    if (!md->mel_filt || !md->rpe) return -1;
    float linc = logf(10000.0f) / (float)(half - 1 > 1 ? half - 1 : 1);
    for (int p = 0; p < 13; p++) {
        float pos = (float)(12 - p);
        for (int i = 0; i < half; i++) {
            float sc = pos * expf(-(float)i * linc);
            md->rpe[(size_t)p * md->a_embd + i]        = sinf(sc);
            md->rpe[(size_t)p * md->a_embd + i + half] = cosf(sc);
        }
    }
    return 0;
}

// One stride-2 3x3 conv (pad 1) + per-channel LayerNorm (weight only, eps 1e-6)
// + ReLU. in is [ci][H][W] planar; returns [co][H'][W'], H'=(H-1)/2+1.
static float *sscp_conv(const struct gguf_tensor *wt, const float *norm_w,
                        const float *in, int ci, int H, int W, int *oh, int *ow) {
    int co = (int)wt->dims[3];
    int Ho = (H - 1) / 2 + 1, Wo = (W - 1) / 2 + 1;
    const float *w = (const float *)wt->data;             // [co][ci][3][3], f32
    float *out = malloc((size_t)co * Ho * Wo * sizeof(float));
    if (!out) return NULL;
    int oc;
    #pragma omp parallel for schedule(static)
    for (oc = 0; oc < co; oc++)
        for (int y = 0; y < Ho; y++)
            for (int x = 0; x < Wo; x++) {
                float s = 0.0f;
                for (int ic = 0; ic < ci; ic++)
                    for (int ky = 0; ky < 3; ky++) {
                        int iy = 2 * y - 1 + ky;
                        if (iy < 0 || iy >= H) continue;
                        for (int kx = 0; kx < 3; kx++) {
                            int ix = 2 * x - 1 + kx;
                            if (ix < 0 || ix >= W) continue;
                            s += in[((size_t)ic * H + iy) * W + ix] *
                                 w[(((size_t)oc * ci + ic) * 3 + ky) * 3 + kx];
                        }
                    }
                out[((size_t)oc * Ho + y) * Wo + x] = s;
            }
    int yx;
    #pragma omp parallel for schedule(static)
    for (yx = 0; yx < Ho * Wo; yx++) {                    // LayerNorm over channels, then ReLU
        float mean = 0.0f, var = 0.0f;
        for (int c = 0; c < co; c++) mean += out[(size_t)c * Ho * Wo + yx];
        mean /= (float)co;
        for (int c = 0; c < co; c++) { float d = out[(size_t)c * Ho * Wo + yx] - mean; var += d * d; }
        var /= (float)co;
        float sc = 1.0f / sqrtf(var + 1e-6f);
        for (int c = 0; c < co; c++) {
            float v = (out[(size_t)c * Ho * Wo + yx] - mean) * sc * norm_w[c];
            out[(size_t)c * Ho * Wo + yx] = v > 0.0f ? v : 0.0f;
        }
    }
    *oh = Ho; *ow = Wo;
    return out;
}

// The conformer front end: the log-mel image through the 2x stride-2
// subsampling conv stack, flattened to the [T][ch*freq] feature rows the blocks
// consume. Split out from the blocks so the host oracle and the GPU encoder run
// on the SAME F — mel and these irregular convs stay on the host either way
// (they are ~2% of the FLOP; the 12 blocks are the bulk the GPU takes over).
static float *a_frontend(struct media *md, const float *mel, int T0, int *T_out, int *n_feat_out) {
    int h1, w1, h2, w2;                                   // subsampling: mel as a 1ch image
    float *c1 = sscp_conv(md->a_conv0, md->a_conv0_n, mel, 1, T0, md->n_mel, &h1, &w1);
    if (!c1) return NULL;
    float *c2 = sscp_conv(md->a_conv1, md->a_conv1_n, c1, (int)md->a_conv0->dims[3], h1, w1, &h2, &w2);
    free(c1);
    if (!c2) return NULL;
    const int T = h2, ch = (int)md->a_conv1->dims[3];     // time x (ch * freq)
    float *F = malloc((size_t)T * (size_t)(ch * w2) * sizeof(float));   // flatten channel-fastest
    if (!F) { free(c2); return NULL; }
    for (int t = 0; t < T; t++)
        for (int f = 0; f < w2; f++)
            for (int c = 0; c < ch; c++)
                F[(size_t)t * ch * w2 + f * ch + c] = c2[((size_t)c * T + t) * w2 + f];
    free(c2);
    *T_out = T; *n_feat_out = ch * w2;
    return F;
}

// matmat through a QAT clippable linear: input clamps into tmp (tmp == X is
// fine — same-index read/write), output clamps in place. Inactive ranges
// (+/-FLT_MAX, i.e. a pre-QAT file) skip both passes and this is plain matmat.
static void matmat_cl(float *out, const struct gguf_tensor *t, const float *X, float *tmp,
                      int k, int m, int T, struct aclamp c) {
    const float *src = X;
    if (c.ilo > -FLT_MAX || c.ihi < FLT_MAX) {
        for (size_t i = 0; i < (size_t)T * k; i++) {
            float v = X[i];
            tmp[i] = v < c.ilo ? c.ilo : v > c.ihi ? c.ihi : v;
        }
        src = tmp;
    }
    matmat(out, t, src, k, m, T);
    if (c.olo > -FLT_MAX || c.ohi < FLT_MAX)
        for (size_t i = 0; i < (size_t)T * m; i++) {
            float v = out[i];
            out[i] = v < c.olo ? c.olo : v > c.ohi ? c.ohi : v;
        }
}

// The 12 conformer blocks (host path / numeric oracle): project F in, run the
// stack, then out-proj -> plain RMS -> the LLM-width projection. F[T][n_feat] is
// the caller's (the GPU encoder consumes the identical buffer). -> [T][n_embd].
static float *a_blocks_host(struct media *md, const float *F, int T, int n_feat, int *n_tokens) {
    const int ne = md->a_embd, nh = md->a_head, dh = ne / nh;
    const float eps = 1e-6f;

    float *X = malloc((size_t)T * ne * sizeof(float));    // projected-in features
    float *H = malloc((size_t)T * ne * sizeof(float));
    float *G = malloc((size_t)T * 4 * ne * sizeof(float));
    float *D = malloc((size_t)T * 2 * ne * sizeof(float));
    float *Q = malloc((size_t)T * ne * sizeof(float));
    float *K = malloc((size_t)T * ne * sizeof(float));
    float *V = malloc((size_t)T * ne * sizeof(float));
    float *P = malloc((size_t)13 * ne * sizeof(float));   // projected RPE
    float *S = malloc((size_t)T * ne * sizeof(float));    // clamp scratch (q/k/v share H)
    if (!X || !H || !G || !D || !Q || !K || !V || !P || !S) goto fail;
    matmat(X, md->a_inp_proj, F, n_feat, ne, T);          // [T][n_feat] -> [T][ne]

    const float q_scale = (1.0f / sqrtf((float)dh)) / logf(2.0f);
    const float k_scale = logf(1.0f + expf(1.0f)) / logf(2.0f);

    for (int L = 0; L < md->a_layer; L++) {
        const struct alayer *a = &md->al[L];

        // ---- FFN 1 (half-step residual) ----
        for (int t = 0; t < T; t++) {
            memcpy(H + (size_t)t * ne, X + (size_t)t * ne, (size_t)ne * 4);
            rmsnorm_w(H + (size_t)t * ne, a->ffn_norm, ne, eps);
        }
        matmat_cl(G, a->ffn_up, H, H, ne, 4 * ne, T, a->c_ffn_up);
        for (size_t i = 0; i < (size_t)T * 4 * ne; i++) G[i] = silu(G[i]);
        matmat_cl(D, a->ffn_down, G, G, 4 * ne, ne, T, a->c_ffn_down);
        for (int t = 0; t < T; t++) {
            rmsnorm_w(D + (size_t)t * ne, a->ffn_post, ne, eps);
            for (int i = 0; i < ne; i++) X[(size_t)t * ne + i] += 0.5f * D[(size_t)t * ne + i];
        }

        // ---- chunked local attention with relative positions ----
        for (int t = 0; t < T; t++) {
            memcpy(H + (size_t)t * ne, X + (size_t)t * ne, (size_t)ne * 4);
            rmsnorm_w(H + (size_t)t * ne, a->attn_pre, ne, eps);
        }
        matmat_cl(Q, a->q, H, S, ne, ne, T, a->c_q);      // S: q/k/v all read H
        matmat_cl(K, a->k, H, S, ne, ne, T, a->c_k);
        matmat_cl(V, a->v, H, S, ne, ne, T, a->c_v);
        matmat(P, a->k_rel, md->rpe, ne, ne, 13);         // k_rel carries no QAT ranges
        for (size_t i = 0; i < (size_t)T * ne; i++) { Q[i] *= q_scale * a->pds[i % dh]; K[i] *= k_scale; }
        int pt;                                           // each pos attends to prev 12 incl. itself
        #pragma omp parallel for schedule(static)
        for (pt = 0; pt < T; pt++) {
            const int t = pt;
            float att[12], *outr = D + (size_t)t * ne;
            int k0 = t - 11 > 0 ? t - 11 : 0;
            for (int h = 0; h < nh; h++) {
                const float *qh = Q + (size_t)t * ne + h * dh;
                float mx = -1e30f;
                for (int gk = k0; gk <= t; gk++) {
                    const float *kh = K + (size_t)gk * ne + h * dh;
                    const float *ph = P + (size_t)(12 - (t - gk)) * ne + h * dh;
                    float s = 0.0f;
                    for (int i = 0; i < dh; i++) s += qh[i] * (kh[i] + ph[i]);
                    s = 50.0f * tanhf(s / 50.0f);
                    att[gk - k0] = s;
                    if (s > mx) mx = s;
                }
                float sum = 0.0f;
                for (int gk = k0; gk <= t; gk++) { att[gk - k0] = expf(att[gk - k0] - mx); sum += att[gk - k0]; }
                for (int i = 0; i < dh; i++) {
                    float o = 0.0f;
                    for (int gk = k0; gk <= t; gk++) o += att[gk - k0] * V[(size_t)gk * ne + h * dh + i];
                    outr[h * dh + i] = o / sum;
                }
            }
        }
        matmat_cl(H, a->o, D, D, ne, ne, T, a->c_o);
        for (int t = 0; t < T; t++) {
            rmsnorm_w(H + (size_t)t * ne, a->attn_post, ne, eps);
            for (int i = 0; i < ne; i++) X[(size_t)t * ne + i] += H[(size_t)t * ne + i];
        }

        // ---- convolution module ----
        for (int t = 0; t < T; t++) {
            memcpy(H + (size_t)t * ne, X + (size_t)t * ne, (size_t)ne * 4);
            rmsnorm_w(H + (size_t)t * ne, a->norm_conv, ne, eps);
        }
        matmat_cl(G, a->pw1, H, H, ne, 2 * ne, T, a->c_pw1);       // -> GLU halves
        for (int t = 0; t < T; t++)
            for (int i = 0; i < ne; i++)
                D[(size_t)t * ne + i] = G[(size_t)t * 2 * ne + i] /
                                        (1.0f + expf(-G[(size_t)t * 2 * ne + ne + i]));
        for (int t = 0; t < T; t++) {                              // causal depthwise, 5 taps
            float *hr = H + (size_t)t * ne;
            for (int i = 0; i < ne; i++) {
                float s = 0.0f;
                for (int j = 0; j < 5; j++) {
                    int ts = t - 4 + j;
                    if (ts >= 0) s += D[(size_t)ts * ne + i] * a->dw[(size_t)i * 5 + j];
                }
                hr[i] = s;
            }
            rmsnorm_w(hr, a->conv_norm, ne, eps);
            for (int i = 0; i < ne; i++) hr[i] = silu(hr[i]);
        }
        matmat_cl(D, a->pw2, H, H, ne, ne, T, a->c_pw2);
        for (size_t i = 0; i < (size_t)T * ne; i++) X[i] += D[i];

        // ---- FFN 2 (half-step residual) ----
        for (int t = 0; t < T; t++) {
            memcpy(H + (size_t)t * ne, X + (size_t)t * ne, (size_t)ne * 4);
            rmsnorm_w(H + (size_t)t * ne, a->ffn_norm1, ne, eps);
        }
        matmat_cl(G, a->ffn_up1, H, H, ne, 4 * ne, T, a->c_ffn_up1);
        for (size_t i = 0; i < (size_t)T * 4 * ne; i++) G[i] = silu(G[i]);
        matmat_cl(D, a->ffn_down1, G, G, 4 * ne, ne, T, a->c_ffn_down1);
        for (int t = 0; t < T; t++) {
            rmsnorm_w(D + (size_t)t * ne, a->ffn_post1, ne, eps);
            for (int i = 0; i < ne; i++) X[(size_t)t * ne + i] += 0.5f * D[(size_t)t * ne + i];
        }

        // ---- layer output norm ----
        for (int t = 0; t < T; t++) rmsnorm_w(X + (size_t)t * ne, a->ln2, ne, eps);
    }

    {                                                     // out proj -> rms norm -> LLM-width proj
        int ao = (int)md->a_out_proj->dims[1];
        float *Y = malloc((size_t)T * ao * sizeof(float));
        float *rows = malloc((size_t)T * md->n_embd * sizeof(float));
        if (!Y || !rows) { free(Y); free(rows); goto fail; }
        matmat(Y, md->a_out_proj, X, ne, ao, T);
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < ao; i++) Y[(size_t)t * ao + i] += md->a_out_proj_b[i];
            rmsnorm_plain(Y + (size_t)t * ao, ao, eps);
        }
        matmat(rows, md->mm_a, Y, ao, md->n_embd, T);
        free(Y);
        free(H); free(G); free(D); free(Q); free(K); free(V); free(P); free(X); free(S);
        *n_tokens = T;
        return rows;
    }
fail:
    free(H); free(G); free(D); free(Q); free(K); free(V); free(P); free(X); free(S);
    return NULL;
}

// ---- audio ------------------------------------------------------------------

static float *uv_embed_audio(struct media *md, const int16_t *pcm, int n_samples, int *n_tokens) {
    const int F = md->frame, ne = md->n_embd;
    int n = n_samples / F;                      // one token per 40 ms frame
    float *out = malloc((size_t)n * ne * sizeof(float));
    float *vec = malloc((size_t)F * sizeof(float));
    if (!out || !vec) { free(out); free(vec); return NULL; }
    for (int t = 0; t < n; t++) {
        for (int i = 0; i < F; i++) vec[i] = (float)pcm[(size_t)t * F + i] / 32768.0f;
        rmsnorm_plain(vec, F, 1e-6f);
        matvec(out + (size_t)t * ne, md->mm_a, vec, F, ne);
    }
    free(vec);
    *n_tokens = n;
    return out;
}

float *media_embed_audio(struct media *md, const int16_t *pcm, int n_samples, int *n_tokens) {
    const int F = md->frame;
    if (md->legacy_a) {
        // E2B/E4B conformer: PCM -> log-mel -> 12 blocks -> LLM-width rows.
        // (the wire already delivers 16 kHz mono PCM; preprocessing is upstream)
        if (n_samples <= 0) { fprintf(stderr, "media: empty audio clip\n"); return NULL; }
        if (n_samples > 30 * 16000) {
            fprintf(stderr, "media: legacy audio is capped at 30 s per clip for now\n");
            return NULL;
        }
        float *x = malloc((size_t)n_samples * sizeof(float));
        if (!x) return NULL;
        for (int i = 0; i < n_samples; i++) x[i] = (float)pcm[i] / 32768.0f;
        int T0;
        float *mel = log_mel(md, x, n_samples, &T0);
        free(x);
        if (!mel) return NULL;
        int T, n_feat;
        float *Fr = a_frontend(md, mel, T0, &T, &n_feat);
        free(mel);
        if (!Fr) return NULL;
        float *rows = a_blocks_gpu(md, Fr, T, n_feat, n_tokens);
        if (rows && getenv("LG_MEDIA_VERIFY")) {        // host blocks as numeric oracle
            int n2 = 0;
            float *ref = a_blocks_host(md, Fr, T, n_feat, &n2);
            verify_rows(md, rows, ref, *n_tokens, n2);
            free(ref);
        }
        if (!rows)                                      // CUDA unusable -> host fallback
            rows = a_blocks_host(md, Fr, T, n_feat, n_tokens);
        free(Fr);
        return rows;
    }
    if (md->legacy_v) {
        fprintf(stderr, "media: E2B/E4B native audio (gemma4a) is off — set LG_GEMMA4A=1 "
                        "to enable it (transcribes with the QAT-era mmprojs; the original "
                        "encoder export confabulates), or use Whisper upstream / the 12B\n");
        return NULL;
    }
    if (n_samples <= 0 || n_samples % F) {
        fprintf(stderr, "media: %d samples is not a multiple of the %d-sample frame\n", n_samples, F);
        return NULL;
    }
    float *rows = uv_embed_audio_gpu(md, pcm, n_samples, n_tokens);
    if (rows && getenv("LG_MEDIA_VERIFY")) {        // host path as numeric oracle
        int n2 = 0;
        float *ref = uv_embed_audio(md, pcm, n_samples, &n2);
        verify_rows(md, rows, ref, *n_tokens, n2);
        free(ref);
    }
    if (rows) return rows;                          // NULL: no GPU / unusable -> host
    return uv_embed_audio(md, pcm, n_samples, n_tokens);
}
