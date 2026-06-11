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

// One conformer block of the legacy (gemma4a) audio encoder — see below.
struct alayer {
    const float *ffn_norm, *ffn_post, *ffn_norm1, *ffn_post1;
    const struct gguf_tensor *ffn_up, *ffn_down, *ffn_up1, *ffn_down1;
    const float *attn_pre, *attn_post, *pds;   // pds: per-dim Q scale [d_head]
    const struct gguf_tensor *q, *k, *v, *o, *k_rel;
    const float *norm_conv, *conv_norm, *ln2, *dw;  // dw: depthwise taps [5][a_embd]
    const struct gguf_tensor *pw1, *pw2;
};

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
    // audio: one linear over raw 640-sample frames (gemma4ua), or the legacy
    // conformer encoder (gemma4a) below
    int frame;                                 // samples per frame (640 = 40 ms @ 16 kHz)
    const struct gguf_tensor *mm_a;            // [frame -> n_embd] / [a_out -> n_embd]
    // ---- legacy audio (gemma4a): mel frontend + 12-block conformer ----------
    int legacy_a;
    int a_embd, a_head, a_layer;               // 1024, 8, 12 (d_head = a_embd/a_head)
    int n_mel;                                 // 128
    const struct gguf_tensor *a_conv0, *a_conv1;   // [3,3,1,c0], [3,3,c0,c1]
    const float *a_conv0_n, *a_conv1_n;            // channel LayerNorm weights
    const struct gguf_tensor *a_inp_proj;          // [flatten -> a_embd]
    const struct gguf_tensor *a_out_proj;          // [a_embd -> a_out]
    const float *a_out_proj_b;
    struct alayer *al;
    float *mel_filt;                               // [n_mel][n_fft/2+1] triangles
    float *rpe;                                    // [13][a_embd] sinusoids
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

static int legacy_audio_open(struct media *md);   // gemma4a, below

struct media *media_open(const char *path, int n_embd) {
    struct gguf_context *ctx = load_gguf(path);
    if (!ctx) return NULL;

    const char *vp = gguf_get_str(ctx, "clip.vision.projector_type", "");
    const char *ap = gguf_get_str(ctx, "clip.audio.projector_type", "");
    int unified  = strcmp(vp, "gemma4uv") == 0 && strcmp(ap, "gemma4ua") == 0;
    int legacy_a = strcmp(ap, "gemma4a") == 0;
    if (!unified && !legacy_a) {
        fprintf(stderr, "media: %s is a '%s'/'%s' mmproj; supported are the "
                        "encoder-free gemma4uv/gemma4ua (12B) and, audio only, "
                        "the legacy gemma4a conformer (E2B/E4B)\n", path, vp, ap);
        free_gguf(ctx);
        return NULL;
    }

    struct media *md = calloc(1, sizeof *md);
    if (!md) { free_gguf(ctx); return NULL; }
    md->ctx = ctx;
    md->n_embd = n_embd;

    if (legacy_a) {
        // E2B/E4B: the 12-block audio conformer works; their 16-block vision
        // transformer (gemma4v) is not implemented — image frames will error.
        md->legacy_a = 1;
        md->frame = 640;             // the wire stays raw 16 kHz PCM either way
        if (legacy_audio_open(md) != 0) { media_free(md); return NULL; }
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
    free(md->al); free(md->mel_filt); free(md->rpe);
    free(md);
}

// ---- vision -----------------------------------------------------------------

float *media_embed_image(struct media *md, const uint8_t *rgb, int w, int h, int *n_tokens) {
    if (md->legacy_a) {
        fprintf(stderr, "media: this mmproj's vision side (gemma4v, a 16-block "
                        "transformer) is not implemented — audio only\n");
        return NULL;
    }
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

// ================  legacy audio: the gemma4a conformer (E2B/E4B)  ============
// The pre-12B design the user's edge app runs on: a log-mel spectrogram, two
// stride-2 convolutions, and 12 conformer blocks (dual half-step SiLU FFNs,
// chunked local attention with sinusoidal relative positions, a gated causal
// depthwise-conv module). Everything below mirrors llama.cpp's
// clip_graph_gemma4a and its gemma4a audio preprocessor. This whole section
// can be deleted the day encoder-free models make E2B/E4B irrelevant.

// out[t][r] = W . x_t for all T rows at once — each (f16) weight row is
// unpacked ONCE and dotted against every timestep while it is hot (the same
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

// RMS norm with a weight, in place over one row.
static void rmsnorm_w(float *x, const float *w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float s = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) x[i] *= s * w[i];
}

static float silu(float x) { return x / (1.0f + expf(-x)); }

#define A_NFFT   512      // FFT size; the 20 ms (320-sample) Hann window is zero-padded to it
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
        for (int i = 0; i < N; i += len) {
            for (int j = 0; j < len / 2; j++) {
                float wr = (float)cos(ang * j), wi = (float)sin(ang * j);
                float ur = a[2*(i+j)],         ui = a[2*(i+j)+1];
                float vr = a[2*(i+j+len/2)]   * wr - a[2*(i+j+len/2)+1] * wi;
                float vi = a[2*(i+j+len/2)]   * wi + a[2*(i+j+len/2)+1] * wr;
                a[2*(i+j)]         = ur + vr; a[2*(i+j)+1]         = ui + vi;
                a[2*(i+j+len/2)]   = ur - vr; a[2*(i+j+len/2)+1]   = ui - vi;
            }
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

static const float *afptr(struct media *md, const char *fmt, int L) {
    char nm[64];
    snprintf(nm, sizeof nm, fmt, L);
    const struct gguf_tensor *t = gguf_find_tensor(md->ctx, nm);
    return t ? (const float *)t->data : NULL;
}
static const struct gguf_tensor *atens(struct media *md, const char *fmt, int L) {
    char nm[64];
    snprintf(nm, sizeof nm, fmt, L);
    return gguf_find_tensor(md->ctx, nm);
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
        a->ffn_norm  = afptr(md, "a.blk.%d.ffn_norm.weight", L);
        a->ffn_post  = afptr(md, "a.blk.%d.ffn_post_norm.weight", L);
        a->ffn_up    = atens(md, "a.blk.%d.ffn_up.weight", L);
        a->ffn_down  = atens(md, "a.blk.%d.ffn_down.weight", L);
        a->ffn_norm1 = afptr(md, "a.blk.%d.ffn_norm_1.weight", L);
        a->ffn_post1 = afptr(md, "a.blk.%d.ffn_post_norm_1.weight", L);
        a->ffn_up1   = atens(md, "a.blk.%d.ffn_up_1.weight", L);
        a->ffn_down1 = atens(md, "a.blk.%d.ffn_down_1.weight", L);
        a->attn_pre  = afptr(md, "a.blk.%d.attn_pre_norm.weight", L);
        a->attn_post = afptr(md, "a.blk.%d.attn_post_norm.weight", L);
        a->pds       = afptr(md, "a.blk.%d.per_dim_scale.weight", L);
        a->q = atens(md, "a.blk.%d.attn_q.weight", L);
        a->k = atens(md, "a.blk.%d.attn_k.weight", L);
        a->v = atens(md, "a.blk.%d.attn_v.weight", L);
        a->o = atens(md, "a.blk.%d.attn_out.weight", L);
        a->k_rel = atens(md, "a.blk.%d.attn_k_rel.weight", L);
        a->norm_conv = afptr(md, "a.blk.%d.norm_conv.weight", L);
        a->conv_norm = afptr(md, "a.blk.%d.conv_norm.weight", L);
        a->ln2       = afptr(md, "a.blk.%d.ln2.weight", L);
        a->dw        = afptr(md, "a.blk.%d.conv_dw.weight", L);
        a->pw1 = atens(md, "a.blk.%d.conv_pw1.weight", L);
        a->pw2 = atens(md, "a.blk.%d.conv_pw2.weight", L);
        if (!a->ffn_norm || !a->ffn_post || !a->ffn_up || !a->ffn_down ||
            !a->ffn_norm1 || !a->ffn_post1 || !a->ffn_up1 || !a->ffn_down1 ||
            !a->attn_pre || !a->attn_post || !a->pds || !a->q || !a->k || !a->v ||
            !a->o || !a->k_rel || !a->norm_conv || !a->conv_norm || !a->ln2 ||
            !a->dw || !a->pw1 || !a->pw2) {
            fprintf(stderr, "media: gemma4a block %d tensors missing\n", L);
            return -1;
        }
    }
    md->mel_filt = mel_filterbank(md->n_mel);
    // Sinusoidal relative positions, table p = distance 12-p (so index 12 is
    // "self"): emb[i] = sin(pos * 10000^(-i/(half-1))), cos in the upper half.
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

// One stride-2 3x3 conv (pad 1) + per-channel LayerNorm (weight only, eps
// 1e-6) + ReLU. in is [ci][H][W] planar; returns [co][H'][W'], H'=(H-1)/2+1.
static float *sscp_conv(const struct gguf_tensor *wt, const float *norm_w,
                        const float *in, int ci, int H, int W, int *oh, int *ow) {
    int co = (int)wt->dims[3];
    int Ho = (H - 1) / 2 + 1, Wo = (W - 1) / 2 + 1;
    const float *w = (const float *)wt->data;             // [co][ci][3][3], f32
    float *out = malloc((size_t)co * Ho * Wo * sizeof(float));
    if (!out) return NULL;
    int oc;                                               // MSVC OpenMP wants it out here
    #pragma omp parallel for schedule(static)
    for (oc = 0; oc < co; oc++) {
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
    }
    // LayerNorm across the co channels at each (y, x), then ReLU
    int yx;
    #pragma omp parallel for schedule(static)
    for (yx = 0; yx < Ho * Wo; yx++) {
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

// The conformer encoder: mel [T0][n_mel] -> tokens [T][n_embd of the LLM].
static float *gemma4a_encode(struct media *md, const float *mel, int T0, int *n_tokens) {
    const int ne = md->a_embd, nh = md->a_head, dh = ne / nh;
    const float eps = 1e-6f;

    // subsampling: treat the spectrogram as a 1-channel [H=T0][W=n_mel] image
    int h1, w1, h2, w2;
    float *c1 = sscp_conv(md->a_conv0, md->a_conv0_n, mel, 1, T0, md->n_mel, &h1, &w1);
    if (!c1) return NULL;
    float *c2 = sscp_conv(md->a_conv1, md->a_conv1_n, c1, (int)md->a_conv0->dims[3], h1, w1, &h2, &w2);
    free(c1);
    if (!c2) return NULL;
    const int T = h2, ch = (int)md->a_conv1->dims[3];     // time x (ch * freq)

    // flatten channel-fastest ([ch*freq, time] with ch in ne0) and project in
    float *X = malloc((size_t)T * ne * sizeof(float));
    float *F = malloc((size_t)T * (size_t)(ch * w2) * sizeof(float));
    if (!X || !F) { free(c2); free(X); free(F); return NULL; }
    for (int t = 0; t < T; t++)
        for (int f = 0; f < w2; f++)
            for (int c = 0; c < ch; c++)
                F[(size_t)t * ch * w2 + f * ch + c] = c2[((size_t)c * T + t) * w2 + f];
    free(c2);
    matmat(X, md->a_inp_proj, F, ch * w2, ne, T);
    free(F);

    // scratch reused across blocks
    float *H  = malloc((size_t)T * ne * sizeof(float));         // normed input
    float *G  = malloc((size_t)T * 4 * ne * sizeof(float));     // widest intermediate
    float *D  = malloc((size_t)T * 2 * ne * sizeof(float));
    float *Q  = malloc((size_t)T * ne * sizeof(float));
    float *K  = malloc((size_t)T * ne * sizeof(float));
    float *V  = malloc((size_t)T * ne * sizeof(float));
    float *P  = malloc((size_t)13 * ne * sizeof(float));        // projected RPE
    if (!H || !G || !D || !Q || !K || !V || !P) goto fail;

    const float q_scale = (1.0f / sqrtf((float)dh)) / logf(2.0f);
    const float k_scale = logf(1.0f + expf(1.0f)) / logf(2.0f);

    for (int L = 0; L < md->a_layer; L++) {
        const struct alayer *a = &md->al[L];

        // ---- FFN 1 (half-step residual) ----
        for (int t = 0; t < T; t++) {
            memcpy(H + (size_t)t * ne, X + (size_t)t * ne, (size_t)ne * 4);
            rmsnorm_w(H + (size_t)t * ne, a->ffn_norm, ne, eps);
        }
        matmat(G, a->ffn_up, H, ne, 4 * ne, T);
        for (size_t i = 0; i < (size_t)T * 4 * ne; i++) G[i] = silu(G[i]);
        matmat(D, a->ffn_down, G, 4 * ne, ne, T);
        for (int t = 0; t < T; t++) {
            rmsnorm_w(D + (size_t)t * ne, a->ffn_post, ne, eps);
            for (int i = 0; i < ne; i++) X[(size_t)t * ne + i] += 0.5f * D[(size_t)t * ne + i];
        }

        // ---- chunked local attention with relative positions ----
        for (int t = 0; t < T; t++) {
            memcpy(H + (size_t)t * ne, X + (size_t)t * ne, (size_t)ne * 4);
            rmsnorm_w(H + (size_t)t * ne, a->attn_pre, ne, eps);
        }
        matmat(Q, a->q, H, ne, ne, T);
        matmat(K, a->k, H, ne, ne, T);
        matmat(V, a->v, H, ne, ne, T);
        matmat(P, a->k_rel, md->rpe, ne, ne, 13);
        for (size_t i = 0; i < (size_t)T * ne; i++) {
            Q[i] *= q_scale * a->pds[i % dh];
            K[i] *= k_scale;
        }
        // each position attends to the previous 12 including itself; the
        // blocked mask in the reference reduces to exactly this window
        int pt;
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
        matmat(H, a->o, D, ne, ne, T);
        for (int t = 0; t < T; t++) {
            rmsnorm_w(H + (size_t)t * ne, a->attn_post, ne, eps);
            for (int i = 0; i < ne; i++) X[(size_t)t * ne + i] += H[(size_t)t * ne + i];
        }

        // ---- convolution module ----
        for (int t = 0; t < T; t++) {
            memcpy(H + (size_t)t * ne, X + (size_t)t * ne, (size_t)ne * 4);
            rmsnorm_w(H + (size_t)t * ne, a->norm_conv, ne, eps);
        }
        matmat(G, a->pw1, H, ne, 2 * ne, T);                       // -> GLU halves
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
        matmat(D, a->pw2, H, ne, ne, T);
        for (size_t i = 0; i < (size_t)T * ne; i++) X[i] += D[i];

        // ---- FFN 2 (half-step residual) ----
        for (int t = 0; t < T; t++) {
            memcpy(H + (size_t)t * ne, X + (size_t)t * ne, (size_t)ne * 4);
            rmsnorm_w(H + (size_t)t * ne, a->ffn_norm1, ne, eps);
        }
        matmat(G, a->ffn_up1, H, ne, 4 * ne, T);
        for (size_t i = 0; i < (size_t)T * 4 * ne; i++) G[i] = silu(G[i]);
        matmat(D, a->ffn_down1, G, 4 * ne, ne, T);
        for (int t = 0; t < T; t++) {
            rmsnorm_w(D + (size_t)t * ne, a->ffn_post1, ne, eps);
            for (int i = 0; i < ne; i++) X[(size_t)t * ne + i] += 0.5f * D[(size_t)t * ne + i];
        }

        // ---- layer output norm ----
        for (int t = 0; t < T; t++) rmsnorm_w(X + (size_t)t * ne, a->ln2, ne, eps);
    }

    // output projection -> plain RMS norm -> the LLM-width projection
    {
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
        free(H); free(G); free(D); free(Q); free(K); free(V); free(P); free(X);
        *n_tokens = T;
        return rows;
    }

fail:
    free(H); free(G); free(D); free(Q); free(K); free(V); free(P); free(X);
    return NULL;
}

// ---- audio ------------------------------------------------------------------

float *media_embed_audio(struct media *md, const int16_t *pcm, int n_samples, int *n_tokens) {
    const int F = md->frame, ne = md->n_embd;
    if (n_samples <= 0 || n_samples % F) {
        fprintf(stderr, "media: %d samples is not a multiple of the %d-sample frame\n", n_samples, F);
        return NULL;
    }
    if (md->legacy_a) {
        if (n_samples > 30 * 16000) {                     // the encoder's context limit
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
        float *rows = gemma4a_encode(md, mel, T0, n_tokens);
        free(mel);
        return rows;
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
