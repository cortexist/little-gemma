// Multimodal embedders for the gemma-4 mmproj files (see media.h).
// Raw data in, embedding rows out — file decoding lives in tools/media_cat.c.
// Everything runs on the host in f32: even the 16-block legacy vision
// transformer is a few hundred GFLOP per image, done once per prompt.
// The math mirrors llama.cpp's clip graphs (gemma4uv/gemma4ua for the 12B,
// gemma4v for E2B/E4B vision) so outputs can be compared against that
// implementation as an oracle — which is how the 12B and gemma4a ports were
// verified to 4-5 decimals, stage by stage.
//
// E2B/E4B AUDIO (gemma4a, a 12-block conformer) was implemented, verified
// against the reference, judged not worth keeping (the production app uses
// Whisper; the 12B's audio works), and removed — git history has it.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "gguf.h"
#include "quant.h"
#include "media.h"

// One block of the legacy (gemma4v) vision transformer.
struct vlayer {
    const float *ln1, *ln2, *attn_post, *ffn_post;   // rms norm weights [768]
    const float *qn, *kn;                            // per-head q/k norms [64]
    const struct gguf_tensor *q, *k, *v, *o;         // [768 -> 768], no biases
    const struct gguf_tensor *gate, *up, *down;      // gated FFN, 768<->3072
};

struct media {
    struct gguf_context *ctx;
    int n_embd;                    // LLM width; every output row is this long
    // vision, unified (gemma4uv): one linear over patch x patch pixel blocks
    int patch;                     // pixels per patch side (48: 16 x merge 3)
    int pos_size;                  // rows in each axis of the position table
    const struct gguf_tensor *v_patch_w;       // [patch*patch*3 -> n_embd]
    const float *v_patch_b;
    const float *vn1w, *vn1b;                  // LayerNorm before the embed
    const float *vn2w, *vn2b;                  // LayerNorm after it
    const float *vn3w, *vn3b;                  // LayerNorm after the pos add
    const float *v_pos;                        // [n_embd, pos_size, 2]: x table, then y
    const struct gguf_tensor *mm_v;            // [enc_width -> n_embd]
    // audio, unified (gemma4ua): one linear over raw 640-sample frames
    int frame;                                 // samples per frame (640 = 40 ms @ 16 kHz)
    const struct gguf_tensor *mm_a;            // [frame -> n_embd]
    // vision, legacy (gemma4v, E2B/E4B): a 16-block ViT + 3x3 average pool
    int legacy_v;
    int v_embd, v_head, v_layer, v_merge;      // 768, 12, 16, 3
    const struct gguf_tensor *v_patch16;       // [16,16,3 -> 768] conv, no bias
    struct vlayer *vl;
};

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
            float s = 0.0f;
            for (int j = 0; j < k; j++) s += buf[j] * x[j];
            out[i] = s;
        }
        free(buf);
    }
}

// out[t][r] = W . x_t for all T rows at once — each (f16) weight row is
// unpacked ONCE and dotted against every position while it is hot (the same
// move as the GPU's batched prefill). X is [T][k] contiguous, out is [T][m].
static void matmat(float *out, const struct gguf_tensor *t, const float *X, int k, int m, int T) {
    const int blck = ggml_blck_size(t->type);
    const size_t row_bytes = (size_t)(k / blck) * ggml_type_size(t->type);
    const unsigned char *base = t->data;
    #pragma omp parallel
    {
        float *buf = malloc((size_t)k * sizeof(float));
        int r;
        #pragma omp for schedule(static)
        for (r = 0; r < m; r++) {
            dequantize_into(t->type, base + (size_t)r * row_bytes, buf, k);
            for (int tt = 0; tt < T; tt++) {
                const float *x = X + (size_t)tt * k;
                float s = 0.0f;
                for (int j = 0; j < k; j++) s += buf[j] * x[j];
                out[(size_t)tt * m + r] = s;
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
        // E2B/E4B: the 16-block vision transformer works; their audio
        // conformer (gemma4a) is deliberately unsupported — audio frames
        // will error (the production stack uses Whisper; the 12B hears).
        md->legacy_v = 1;
        md->frame = 640;
        if (legacy_vision_open(md) != 0) { media_free(md); return NULL; }
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
    free_gguf(md->ctx);
    free(md->vl);
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

    float *out = malloc((size_t)n * ne * sizeof(float));
    float *vec = malloc((size_t)pin * sizeof(float));
    float *e   = malloc((size_t)ne * sizeof(float));
    if (!out || !vec || !e) { free(out); free(vec); free(e); return NULL; }

    for (int p = 0; p < n; p++) {
        int col = p % n_cols, row = p / n_cols;
        // flatten the patch the way im2col over a planar [W,H,3] image does:
        // channel-major, row-major pixels within the channel, in [0,1]
        for (int c = 0; c < 3; c++)
            for (int ky = 0; ky < P; ky++)
                for (int kx = 0; kx < P; kx++)
                    vec[(c * P + ky) * P + kx] =
                        (float)rgb[3 * ((size_t)(row * P + ky) * w + (col * P + kx)) + c] / 255.0f;

        layernorm(vec, md->vn1w, md->vn1b, pin, 1e-5f);
        matvec(e, md->v_patch_w, vec, pin, ne);
        for (int i = 0; i < ne; i++) e[i] += md->v_patch_b[i];
        layernorm(e, md->vn2w, md->vn2b, ne, 1e-5f);
        const float *ex = md->v_pos + (size_t)col * ne;                       // x table
        const float *ey = md->v_pos + ((size_t)md->pos_size + row) * ne;      // y table
        for (int i = 0; i < ne; i++) e[i] += ex[i] + ey[i];
        layernorm(e, md->vn3w, md->vn3b, ne, 1e-5f);
        rmsnorm_plain(e, ne, 1e-6f);
        matvec(out + (size_t)p * ne, md->mm_v, e, ne, ne);
    }
    free(vec); free(e);
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

    for (int L = 0; L < md->v_layer; L++) {
        const struct vlayer *v = &md->vl[L];

        // ---- attention ----
        for (int p = 0; p < np; p++) {
            memcpy(H + (size_t)p * ne, X + (size_t)p * ne, (size_t)ne * 4);
            rmsnorm_w(H + (size_t)p * ne, v->ln1, ne, eps);
        }
        matmat(Q, v->q, H, ne, ne, np);
        matmat(K, v->k, H, ne, ne, np);
        matmat(V, v->v, H, ne, ne, np);
        for (int p = 0; p < np; p++)
            for (int hh = 0; hh < nh; hh++) {
                rmsnorm_w(Q + (size_t)p * ne + hh * dh, v->qn, dh, eps);
                rmsnorm_w(K + (size_t)p * ne + hh * dh, v->kn, dh, eps);
                rmsnorm_plain(V + (size_t)p * ne + hh * dh, dh, eps);
            }
        rope2d(Q, np, ne, nh, n_cols, 100.0f);
        rope2d(K, np, ne, nh, n_cols, 100.0f);
        {
            int p;
            #pragma omp parallel for schedule(static)
            for (p = 0; p < np; p++) {                      // full attention, scale 1.0
                float *att = malloc((size_t)np * sizeof(float));
                for (int hh = 0; hh < nh; hh++) {
                    const float *qh = Q + (size_t)p * ne + hh * dh;
                    float mx = -1e30f;
                    for (int kk = 0; kk < np; kk++) {
                        const float *kh = K + (size_t)kk * ne + hh * dh;
                        float s = 0.0f;
                        for (int i = 0; i < dh; i++) s += qh[i] * kh[i];
                        att[kk] = s;
                        if (s > mx) mx = s;
                    }
                    float sum = 0.0f;
                    for (int kk = 0; kk < np; kk++) { att[kk] = expf(att[kk] - mx); sum += att[kk]; }
                    float *outr = D + (size_t)p * ne + hh * dh;
                    for (int i = 0; i < dh; i++) {
                        float o = 0.0f;
                        for (int kk = 0; kk < np; kk++) o += att[kk] * V[(size_t)kk * ne + hh * dh + i];
                        outr[i] = o / sum;
                    }
                }
                free(att);
            }
        }
        matmat(H, v->o, D, ne, ne, np);
        for (int p = 0; p < np; p++) {
            rmsnorm_w(H + (size_t)p * ne, v->attn_post, ne, eps);
            for (int i = 0; i < ne; i++) X[(size_t)p * ne + i] += H[(size_t)p * ne + i];
        }

        // ---- gated FFN: gelu_quick(gate) * up -> down ----
        for (int p = 0; p < np; p++) {
            memcpy(H + (size_t)p * ne, X + (size_t)p * ne, (size_t)ne * 4);
            rmsnorm_w(H + (size_t)p * ne, v->ln2, ne, eps);
        }
        matmat(G, v->gate, H, ne, 4 * ne, np);
        matmat(U, v->up, H, ne, 4 * ne, np);
        for (size_t i = 0; i < (size_t)np * 4 * ne; i++) G[i] = gelu_quick(G[i]) * U[i];
        matmat(D, v->down, G, 4 * ne, ne, np);
        for (int p = 0; p < np; p++) {
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

float *media_embed_image(struct media *md, const uint8_t *rgb, int w, int h, int *n_tokens) {
    int P = media_patch(md);
    if (w <= 0 || h <= 0 || w % P || h % P) {
        fprintf(stderr, "media: image %dx%d is not a multiple of the %d-pixel patch\n", w, h, P);
        return NULL;
    }
    return md->legacy_v ? v_embed_image(md, rgb, w, h, n_tokens)
                        : uv_embed_image(md, rgb, w, h, n_tokens);
}

// ---- audio ------------------------------------------------------------------

float *media_embed_audio(struct media *md, const int16_t *pcm, int n_samples, int *n_tokens) {
    const int F = md->frame, ne = md->n_embd;
    if (md->legacy_v) {
        fprintf(stderr, "media: E2B/E4B audio (gemma4a) is not supported — "
                        "use Whisper upstream, or the 12B, which hears\n");
        return NULL;
    }
    if (n_samples <= 0 || n_samples % F) {
        fprintf(stderr, "media: %d samples is not a multiple of the %d-sample frame\n", n_samples, F);
        return NULL;
    }
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
