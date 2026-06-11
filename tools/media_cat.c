// media_cat: the media half of socket_cat. Decodes image and audio FILES into
// the raw, model-ready data little-gemma's socket protocol carries — u8 RGB
// resized so both sides are multiples of the 48-pixel patch (40..280 patches
// total, aspect preserved), and mono 16 kHz s16 PCM padded to whole 640-sample
// frames — sends them as typed length-prefixed frames followed by the question
// line, and streams the reply to stdout until the turn ends.
//
//     media_cat <socket> [-i photo.jpg | -a clip.wav | -t "text"]... "question"
//
// -t sends a text chunk that lands between media spans in the same turn —
// the shape a video takes: -t "0:01" -i f1.jpg -t "0:02" -i f2.jpg ...
//
// The split is deliberate: the model runner handles only what the model's own
// tensors define; everything about FILE formats — JPEG entropy coding, WAV
// chunk soup, resampling, resize filters — lives here, where it can grow
// (video, capture, tiling) without the runner changing. Junk pixels are junk
// whether or not a valid JPEG wrapped them; the runner only checks geometry.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#ifdef _WIN32
#include <winsock2.h>   // must precede windows.h
#include <windows.h>
#include <afunix.h>     // AF_UNIX on Windows 10 1803+
typedef SOCKET sock_t;
#define sock_close closesocket
#define SHUT_SEND SD_SEND
static int sock_init(void) { WSADATA w; return WSAStartup(MAKEWORD(2, 2), &w); }
#else
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <signal.h>
typedef int sock_t;
#define INVALID_SOCKET (-1)
#define sock_close close
#define SHUT_SEND SHUT_WR
static int sock_init(void) { signal(SIGPIPE, SIG_IGN); return 0; }
#endif

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_NO_FAILURE_STRINGS
#include "stb_image.h"

#include "media.h"   // the frame format + the geometry the runner expects

// The gemma4uv geometry this tool prepares data for. (If a future mmproj
// changes these, make them flags; the runner derives its own from tensors.)
#define PATCH    48
#define MIN_TOK  40
#define MAX_TOK  280
#define RATE     16000
#define FRAME    640

// ---- image: decode -> resize to patch-multiple dims -> u8 RGB ---------------

// Pick the resized (W, H): each side a multiple of the patch size, total pixels
// in [MIN_TOK, MAX_TOK] patches' worth, aspect ratio preserved (the qwen-style
// rounding the reference uses — round first, then sqrt-rescale if out of range).
static void target_size(int w, int h, int *ow, int *oh) {
    const float fa = (float)PATCH;
    const long long min_px = (long long)MIN_TOK * PATCH * PATCH;
    const long long max_px = (long long)MAX_TOK * PATCH * PATCH;
    int hb = (int)roundf((float)h / fa) * PATCH;
    int wb = (int)roundf((float)w / fa) * PATCH;
    if (hb < PATCH) hb = PATCH;
    if (wb < PATCH) wb = PATCH;
    if ((long long)hb * wb > max_px) {
        float beta = sqrtf((float)h * (float)w / (float)max_px);
        hb = (int)floorf((float)h / beta / fa) * PATCH;
        wb = (int)floorf((float)w / beta / fa) * PATCH;
        if (hb < PATCH) hb = PATCH;
        if (wb < PATCH) wb = PATCH;
    } else if ((long long)hb * wb < min_px) {
        float beta = sqrtf((float)min_px / ((float)h * (float)w));
        hb = (int)ceilf((float)h * beta / fa) * PATCH;
        wb = (int)ceilf((float)w * beta / fa) * PATCH;
    }
    *ow = wb; *oh = hb;
}

// Bilinear resize on u8 RGB, corner-aligned — arithmetic identical to the
// reference implementation so the pixels (and so the embeddings) match.
static uint8_t *resize_bilinear(const uint8_t *src, int sw, int sh, int dw, int dh) {
    uint8_t *dst = malloc((size_t)3 * dw * dh);
    if (!dst) return NULL;
    float xr = dw > 1 ? (float)(sw - 1) / (float)(dw - 1) : 0.0f;
    float yr = dh > 1 ? (float)(sh - 1) / (float)(dh - 1) : 0.0f;
    for (int y = 0; y < dh; y++) {
        for (int x = 0; x < dw; x++) {
            float px = x * xr, py = y * yr;
            int x0 = (int)px; if (x0 > sw - 1) x0 = sw - 1;
            int y0 = (int)py; if (y0 > sh - 1) y0 = sh - 1;
            int x1 = x0 + 1 < sw ? x0 + 1 : sw - 1;
            int y1 = y0 + 1 < sh ? y0 + 1 : sh - 1;
            float xf = px - x0, yf = py - y0;
            for (int c = 0; c < 3; c++) {
                float t00 = src[3 * (y0 * sw + x0) + c], t01 = src[3 * (y0 * sw + x1) + c];
                float t10 = src[3 * (y1 * sw + x0) + c], t11 = src[3 * (y1 * sw + x1) + c];
                float top = t00 + (t01 - t00) * xf, bot = t10 + (t11 - t10) * xf;
                dst[3 * (y * dw + x) + c] = (uint8_t)(top + (bot - top) * yf);
            }
        }
    }
    return dst;
}

static uint8_t *prep_image(const char *path, int *ow, int *oh) {
    int sw, sh, comp;
    uint8_t *rgb = stbi_load(path, &sw, &sh, &comp, 3);
    if (!rgb) { fprintf(stderr, "media_cat: cannot decode image %s\n", path); return NULL; }
    int dw, dh;
    target_size(sw, sh, &dw, &dh);
    uint8_t *res = resize_bilinear(rgb, sw, sh, dw, dh);
    stbi_image_free(rgb);
    *ow = dw; *oh = dh;
    return res;
}

// ---- audio: WAV -> mono 16 kHz s16, padded to whole frames -------------------

// Minimal RIFF/WAVE reader: 16-bit PCM, any rate, any channel count.
static int16_t *prep_audio(const char *path, int *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "media_cat: cannot open %s\n", path); return NULL; }
    unsigned char hdr[12];
    if (fread(hdr, 1, 12, f) != 12 || memcmp(hdr, "RIFF", 4) || memcmp(hdr + 8, "WAVE", 4)) {
        fprintf(stderr, "media_cat: %s is not a WAV file\n", path); fclose(f); return NULL;
    }
    int fmt = 0, channels = 0, rate = 0, bits = 0;
    int16_t *pcm = NULL;
    size_t n_pcm = 0;
    unsigned char ch[8];
    while (fread(ch, 1, 8, f) == 8) {
        uint32_t sz; memcpy(&sz, ch + 4, 4);
        if (!memcmp(ch, "fmt ", 4)) {
            unsigned char b[16];
            if (sz < 16 || fread(b, 1, 16, f) != 16) break;
            fmt = b[0] | (b[1] << 8); channels = b[2] | (b[3] << 8);
            memcpy(&rate, b + 4, 4); bits = b[14] | (b[15] << 8);
            if (sz > 16) fseek(f, sz - 16, SEEK_CUR);
        } else if (!memcmp(ch, "data", 4)) {
            pcm = malloc(sz);
            if (!pcm || fread(pcm, 1, sz, f) != sz) { free(pcm); pcm = NULL; break; }
            n_pcm = sz / 2;
        } else {
            fseek(f, sz + (sz & 1), SEEK_CUR);
        }
    }
    fclose(f);
    if (!pcm || fmt != 1 || bits != 16 || channels < 1 || rate <= 0) {
        fprintf(stderr, "media_cat: %s: need 16-bit PCM WAV (fmt %d, %d-bit)\n", path, fmt, bits);
        free(pcm); return NULL;
    }

    // downmix to mono f32, resample to 16 kHz (linear), quantize back to s16
    // padded up to a whole number of frames so the runner never sees a tail
    size_t n_in = n_pcm / (size_t)channels;
    float *mono = malloc(n_in * sizeof(float));
    if (!mono) { free(pcm); return NULL; }
    for (size_t i = 0; i < n_in; i++) {
        float s = 0.0f;
        for (int c = 0; c < channels; c++) s += (float)pcm[i * channels + c];
        mono[i] = s / (float)channels;
    }
    free(pcm);

    size_t n16 = rate == RATE ? n_in : (size_t)((double)n_in * RATE / rate);
    size_t n = (n16 + FRAME - 1) / FRAME * FRAME;
    int16_t *out = calloc(n, sizeof(int16_t));
    if (!out) { free(mono); return NULL; }
    for (size_t i = 0; i < n16; i++) {
        double p = (double)i * rate / RATE;
        size_t i0 = (size_t)p;
        size_t i1 = i0 + 1 < n_in ? i0 + 1 : n_in - 1;
        float v = mono[i0] + (mono[i1] - mono[i0]) * (float)(p - (double)i0);
        out[i] = (int16_t)lrintf(v);
    }
    free(mono);
    *n_out = (int)n;
    return out;
}

// ---- the wire ----------------------------------------------------------------

static int send_all(sock_t s, const void *buf, size_t n) {
    const char *p = buf;
    while (n > 0) {
        int k = (int)send(s, p, (int)(n > 1 << 20 ? 1 << 20 : n), 0);
        if (k <= 0) return -1;
        p += k; n -= (size_t)k;
    }
    return 0;
}

static int send_frame(sock_t s, uint8_t type, int w, int h, const void *payload, uint32_t len) {
    uint8_t hdr[MEDIA_FRAME_HDR] = { MEDIA_FRAME_MAGIC, type };
    uint16_t w16 = (uint16_t)w, h16 = (uint16_t)h;
    memcpy(hdr + 2, &w16, 2);
    memcpy(hdr + 4, &h16, 2);
    memcpy(hdr + 6, &len, 4);
    if (send_all(s, hdr, sizeof hdr) != 0) return -1;
    return send_all(s, payload, len);
}

int main(int argc, char **argv) {
    const char *spath = argc > 1 ? argv[1] : NULL;
    const char *question = NULL;
    if (!spath || argc < 3) {
        fprintf(stderr, "usage: %s <socket> [-i image]... [-a audio]... \"question\"\n", argv[0]);
        return 1;
    }

    struct sockaddr_un sa;
    memset(&sa, 0, sizeof sa);
    sa.sun_family = AF_UNIX;
    sock_t s;
    if (sock_init() != 0 || strlen(spath) >= sizeof sa.sun_path ||
        (s = socket(AF_UNIX, SOCK_STREAM, 0)) == INVALID_SOCKET) {
        fprintf(stderr, "socket setup failed (path too long?)\n"); return 1;
    }
    strcpy(sa.sun_path, spath);
    if (connect(s, (struct sockaddr *)&sa, sizeof sa) != 0) {
        fprintf(stderr, "connect to %s failed\n", spath); return 1;
    }

    // decode and send each media argument, in order, then the question line
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-i") && i + 1 < argc) {
            int w, h;
            uint8_t *rgb = prep_image(argv[++i], &w, &h);
            if (!rgb) return 1;
            fprintf(stderr, "media_cat: %s -> %dx%d (%d tokens)\n", argv[i], w, h, (w / PATCH) * (h / PATCH));
            if (send_frame(s, MEDIA_FRAME_IMAGE, w, h, rgb, (uint32_t)(3 * w * h)) != 0) return 1;
            free(rgb);
        } else if (!strcmp(argv[i], "-a") && i + 1 < argc) {
            int n;
            int16_t *pcm = prep_audio(argv[++i], &n);
            if (!pcm) return 1;
            fprintf(stderr, "media_cat: %s -> %.2fs (%d tokens)\n", argv[i], n / (double)RATE, n / FRAME);
            if (send_frame(s, MEDIA_FRAME_AUDIO, 0, 0, pcm, (uint32_t)(2 * n)) != 0) return 1;
            free(pcm);
        } else if (!strcmp(argv[i], "-t") && i + 1 < argc) {
            const char *t = argv[++i];
            if (send_frame(s, MEDIA_FRAME_TEXT, 0, 0, t, (uint32_t)strlen(t)) != 0) return 1;
        } else {
            question = argv[i];
        }
    }
    if (!question) { fprintf(stderr, "media_cat: no question given\n"); return 1; }
    char line[8192];
    snprintf(line, sizeof line, "%s\n", question);
    if (send_all(s, line, strlen(line)) != 0) { fprintf(stderr, "send failed\n"); return 1; }
    shutdown(s, SHUT_SEND);

    // stream the reply until the server closes (it finishes the turn in flight)
    char buf[4096];
    int k;
    while ((k = (int)recv(s, buf, sizeof buf, 0)) > 0) {
        fwrite(buf, 1, (size_t)k, stdout);
        fflush(stdout);
    }
    putchar('\n');
    sock_close(s);
    return 0;
}
