// Multimodal embedders for the gemma-4-12B "unified" mmproj (see media.h).
// Raw data in, embedding rows out — file decoding lives in tools/media_cat.c.
// Everything runs on the host in f32: the whole projector is a handful of
// matmuls over at most a few hundred rows, dwarfed by a single LLM layer.
// The math mirrors llama.cpp's clip_graph_gemma4uv / gemma4ua exactly —
// pytorch LayerNorm (eps 1e-5, hardcoded there too) around the patch embed,
// a plain RMS norm (eps 1e-6) before the final projection — so outputs can
// be compared against that implementation as an oracle.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "gguf.h"
#include "quant.h"
#include "media.h"

struct media {
    struct gguf_context *ctx;
    int n_embd;                    // LLM width; every output row is this long
    // vision (gemma4uv): one linear over patch x patch pixel blocks
    int patch;                     // pixels per patch side (48: 16 x merge 3)
    int pos_size;                  // rows in each axis of the position table
    const struct gguf_tensor *v_patch_w;       // [patch*patch*3 -> n_embd]
    const float *v_patch_b;
    const float *vn1w, *vn1b;                  // LayerNorm before the embed
    const float *vn2w, *vn2b;                  // LayerNorm after it
    const float *vn3w, *vn3b;                  // LayerNorm after the pos add
    const float *v_pos;                        // [n_embd, pos_size, 2]: x table, then y
    const struct gguf_tensor *mm_v;            // [n_embd -> n_embd]
    // audio (gemma4ua): one linear over raw 640-sample frames
    int frame;                                 // samples per frame (640 = 40 ms @ 16 kHz)
    const struct gguf_tensor *mm_a;            // [frame -> n_embd]
};

int media_patch(const struct media *md) { return md->patch; }
int media_frame(const struct media *md) { return md->frame; }

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

// Plain RMS norm (no weight), in place — the pre-projection norm.
static void rmsnorm_plain(float *x, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float s = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) x[i] *= s;
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
            float s = 0.0f;
            for (int j = 0; j < k; j++) s += buf[j] * x[j];
            out[i] = s;
        }
        free(buf);
    }
}

// ---- open / free ------------------------------------------------------------

static const float *fptr(const struct gguf_context *ctx, const char *name) {
    const struct gguf_tensor *t = gguf_find_tensor(ctx, name);
    return t ? (const float *)t->data : NULL;
}

struct media *media_open(const char *path, int n_embd) {
    struct gguf_context *ctx = load_gguf(path);
    if (!ctx) return NULL;

    const char *vp = gguf_get_str(ctx, "clip.vision.projector_type", "");
    const char *ap = gguf_get_str(ctx, "clip.audio.projector_type", "");
    if (strcmp(vp, "gemma4uv") != 0 || strcmp(ap, "gemma4ua") != 0) {
        // The E2B/E4B files carry full encoder stacks (gemma4v: a 16-block
        // vision transformer; gemma4a: a 12-block conformer) — the legacy
        // path, not implemented here.
        fprintf(stderr, "media: %s is a '%s'/'%s' mmproj; only the encoder-free "
                        "gemma4uv/gemma4ua (12B) design is supported\n", path, vp, ap);
        free_gguf(ctx);
        return NULL;
    }

    struct media *md = calloc(1, sizeof *md);
    if (!md) { free_gguf(ctx); return NULL; }
    md->ctx = ctx;
    md->n_embd = n_embd;
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
    free(md);
}

// ---- vision -----------------------------------------------------------------

float *media_embed_image(struct media *md, const uint8_t *rgb, int w, int h, int *n_tokens) {
    const int P = md->patch, pin = P * P * 3, ne = md->n_embd;
    if (w <= 0 || h <= 0 || w % P || h % P) {
        fprintf(stderr, "media: image %dx%d is not a multiple of the %d-pixel patch\n", w, h, P);
        return NULL;
    }
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

// ---- audio ------------------------------------------------------------------

float *media_embed_audio(struct media *md, const int16_t *pcm, int n_samples, int *n_tokens) {
    const int F = md->frame, ne = md->n_embd;
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
