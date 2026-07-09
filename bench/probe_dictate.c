/* probe_dictate.c - ttfc_dictate's duplex sibling: a paced dictation client
 * that can interleave LISTENER PROBES ('P' frames) with the 'T' commits, for
 * the two things the engine must prove:
 *   1. INVISIBILITY: a probed session's reply is byte-identical to an
 *      unprobed one. Reply text goes to stdout RAW (probe envelopes stripped),
 *      so `diff` between runs IS the gate.
 *   2. COST: each probe's wall latency (send -> "<probe|>\n" complete) prints
 *      to stderr, measured while the dictation keeps streaming - the live
 *      cadence a voice client would see.
 *
 *   probe_dictate <sock> <line_file> <wpf> <fps> [probe_every] [probe_gen] [suffix_file]
 *
 * wpf words per 'T' frame at fps frames/sec (ttfc_dictate's pacing); after
 * every probe_every-th frame (0 = never) a probe rides along, carrying the
 * suffix (a built-in listener-check line, or suffix_file's contents) and
 * asking for up to probe_gen tokens. The socket drains non-blocking between
 * frames, like a real voice client's mic loop. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef _WIN32
#include <winsock2.h>
#include <afunix.h>
#include <windows.h>
typedef SOCKET sock_t;
#define sock_close closesocket
static void msleep(int ms) { Sleep(ms); }
static int  wouldblock(void) { return WSAGetLastError() == WSAEWOULDBLOCK; }
static void set_nonblock(sock_t s, int nb) { u_long v = (u_long)nb; ioctlsocket(s, FIONBIO, &v); }
static double now_sec(void) {
    static LARGE_INTEGER f;
    LARGE_INTEGER c;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / f.QuadPart;
}
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "winmm.lib")
#else
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
typedef int sock_t;
#define INVALID_SOCKET (-1)
#define sock_close close
static void msleep(int ms) { usleep(ms * 1000); }
static int  wouldblock(void) { return errno == EAGAIN || errno == EWOULDBLOCK; }
static void set_nonblock(sock_t s, int nb) {
    int fl = fcntl(s, F_GETFL, 0);
    fcntl(s, F_SETFL, nb ? fl | O_NONBLOCK : fl & ~O_NONBLOCK);
}
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * ts.tv_nsec;
}
#endif

#define FRAME_MAGIC 0x01

static const char *g_suffix =
    "\n(brief listener check - I am still mid-request, keep listening; emit "
    "exactly one backchannel tag: [[nod]] if you follow me so far, [[mhmm]] to "
    "acknowledge me, [[answer]] if my request is already complete, [[quiet]] "
    "otherwise)";

static int send_all(sock_t s, const void *buf, size_t n) {
    const char *p = buf;
    while (n > 0) {
        int k = (int)send(s, p, (int)n, 0);
        if (k <= 0) {
            if (k < 0 && wouldblock()) { msleep(1); continue; }
            return -1;
        }
        p += k; n -= (size_t)k;
    }
    return 0;
}
static int send_frame(sock_t s, uint8_t type, uint16_t w, uint16_t h, const void *payload, uint32_t len) {
    uint8_t hdr[10] = { FRAME_MAGIC, type };
    memcpy(hdr + 2, &w, 2);
    memcpy(hdr + 4, &h, 2);
    memcpy(hdr + 6, &len, 4);
    if (send_all(s, hdr, sizeof hdr) != 0) return -1;
    return send_all(s, payload, len);
}

/* Incremental "<|probe>verdict<probe|>\n" extractor: envelope bytes never
 * reach `out`, completed verdicts print to stderr with their latency. A held
 * partial marker carries across reads; neither marker contains '<' past its
 * first byte, so flush-then-rematch-current is exact. */
static struct { int m, cap, cm, vn; char v[512]; } g_pf;
static int    g_probes_sent = 0, g_verdicts = 0;
static double g_probe_t0 = 0, g_lat_sum = 0, g_lat_max = 0;

static int pf_feed(const char *in, int n, char *out) {
    static const char PO[] = "<|probe>", PC[] = "<probe|>\n";
    int on = 0;
    for (int i = 0; i < n; i++) {
        char c = in[i];
        if (!g_pf.cap) {
            if (c == PO[g_pf.m]) {
                if (!PO[++g_pf.m]) { g_pf.m = 0; g_pf.cap = 1; g_pf.cm = 0; g_pf.vn = 0; }
            } else {
                for (int j = 0; j < g_pf.m; j++) out[on++] = PO[j];
                g_pf.m = c == PO[0] ? 1 : 0;
                if (!g_pf.m) out[on++] = c;
            }
        } else {
            if (c == PC[g_pf.cm]) {
                if (!PC[++g_pf.cm]) {
                    double lat = now_sec() - g_probe_t0;
                    g_pf.v[g_pf.vn] = 0;
                    g_verdicts++;
                    g_lat_sum += lat;
                    if (lat > g_lat_max) g_lat_max = lat;
                    fprintf(stderr, "probe %d: %.3fs -> '%s'\n", g_verdicts, lat, g_pf.v);
                    g_pf.cap = 0; g_pf.cm = 0; g_pf.vn = 0;
                }
            } else {
                for (int j = 0; j < g_pf.cm && g_pf.vn < (int)sizeof g_pf.v - 1; j++) g_pf.v[g_pf.vn++] = PC[j];
                g_pf.cm = c == PC[0] ? 1 : 0;
                if (!g_pf.cm && g_pf.vn < (int)sizeof g_pf.v - 1) g_pf.v[g_pf.vn++] = c;
            }
        }
    }
    return on;
}

/* Drain whatever the server has sent; filtered reply bytes go to stdout.
 * Returns the count of "<turn|>" ends seen (after filtering). */
static int drain(sock_t s, int *tstate) {
    static const char T[] = "<turn|>";
    char in[4096], out[4096 + 16];
    int ends = 0;
    for (;;) {
        int k = (int)recv(s, in, sizeof in, 0);
        if (k <= 0) break;
        int on = pf_feed(in, k, out);
        fwrite(out, 1, (size_t)on, stdout);
        for (int i = 0; i < on; i++) {
            if (out[i] == T[*tstate]) { if (!T[++*tstate]) { ends++; *tstate = 0; } }
            else *tstate = out[i] == T[0] ? 1 : 0;
        }
    }
    fflush(stdout);
    return ends;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <sock> <line_file> <wpf> <fps> [probe_every=0] [probe_gen=6] [suffix_file|-] [img_after=0]\n"
                        "  img_after N: a synthetic 96x96 image frame rides after the Nth 'T' frame\n"
                        "  (needs -mm on the server) - the media leg of the invisibility gate\n", argv[0]);
        return 1;
    }
    int wpf = atoi(argv[3]);
    double fps = atof(argv[4]);
    int probe_every = argc > 5 ? atoi(argv[5]) : 0;
    int probe_gen = argc > 6 ? atoi(argv[6]) : 6;
    int img_after = argc > 8 ? atoi(argv[8]) : 0;
    static char sufbuf[4096];
    if (argc > 7 && strcmp(argv[7], "-") != 0) {
        FILE *sf = fopen(argv[7], "rb");
        if (!sf) { fprintf(stderr, "cannot open %s\n", argv[7]); return 1; }
        size_t sn = fread(sufbuf, 1, sizeof sufbuf - 1, sf);
        fclose(sf);
        while (sn && (sufbuf[sn - 1] == '\n' || sufbuf[sn - 1] == '\r')) sn--;
        sufbuf[sn] = 0;
        g_suffix = sufbuf;
    }

    FILE *f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    static char line[1 << 20];
    size_t n = fread(line, 1, sizeof line - 1, f);
    fclose(f);
    line[n] = 0;
    while (n && (line[n - 1] == '\n' || line[n - 1] == '\r' || line[n - 1] == ' ')) line[--n] = 0;

#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
    timeBeginPeriod(1);
#endif
    struct sockaddr_un sa;
    memset(&sa, 0, sizeof sa);
    sa.sun_family = AF_UNIX;
    if (strlen(argv[1]) >= sizeof sa.sun_path) { fprintf(stderr, "socket path too long\n"); return 1; }
    strcpy(sa.sun_path, argv[1]);
    sock_t s = socket(AF_UNIX, SOCK_STREAM, 0);
    if (s == INVALID_SOCKET || connect(s, (struct sockaddr *)&sa, sizeof sa) != 0) {
        fprintf(stderr, "connect to %s failed\n", argv[1]);
        return 1;
    }
    set_nonblock(s, 1);

    int tstate = 0, frames = 0;
    double t0 = now_sec();
    if (wpf == 0) {
        if (send_all(s, line, strlen(line)) != 0 || send_all(s, "\n", 1) != 0) return 1;
    } else {
        char *w = line, frame[8192];
        while (*w) {
            char *g = w;
            for (int k = 0; k < wpf && *w; k++) {
                while (*w && *w != ' ') w++;
                if (*w && k + 1 < wpf) w++;
            }
            int glen = (int)(w - g), off = g != line;      /* leading space on later groups */
            if (off) frame[0] = ' ';
            memcpy(frame + off, g, (size_t)glen);
            if (send_frame(s, 'T', 0, 0, frame, (uint32_t)(glen + off)) != 0) return 1;
            frames++;
            if (img_after > 0 && frames == img_after) {    /* the media leg: a deterministic */
                static uint8_t rgb[96 * 96 * 3];           /* pattern, same bytes every run  */
                for (int i = 0; i < (int)sizeof rgb; i++) rgb[i] = (uint8_t)(i * 7 + (i >> 8) * 13);
                if (send_frame(s, 'I', 96, 96, rgb, sizeof rgb) != 0) return 1;
            }
            if (probe_every > 0 && frames % probe_every == 0 && !g_pf.cap && g_probes_sent == g_verdicts) {
                g_probe_t0 = now_sec();
                g_probes_sent++;
                if (send_frame(s, 'P', (uint16_t)probe_gen, 0, g_suffix, (uint32_t)strlen(g_suffix)) != 0) return 1;
            }
            if (fps > 0) msleep((int)(1000.0 / fps));
            drain(s, &tstate);
            while (*w == ' ') w++;
        }
        if (send_all(s, "\n", 1) != 0) return 1;
    }
    double t_line = now_sec();

    int done = 0;
    double t_first = 0;
    while (!done) {
        char in[4096], out[4096 + 16];
        int k = (int)recv(s, in, sizeof in, 0);
        if (k < 0 && wouldblock()) { msleep(2); continue; }
        if (k <= 0) break;
        if (!t_first) t_first = now_sec();
        int on = pf_feed(in, k, out);
        fwrite(out, 1, (size_t)on, stdout);
        static const char T[] = "<turn|>";
        for (int i = 0; i < on; i++) {
            if (out[i] == T[tstate]) { if (!T[++tstate]) { done = 1; tstate = 0; } }
            else tstate = out[i] == T[0] ? 1 : 0;
        }
        fflush(stdout);
    }
    putchar('\n');
    fflush(stdout);
    sock_close(s);

    fprintf(stderr, "ttft %.3fs | send window %.1fs | %d frames\n",
            t_first ? t_first - t_line : -1, t_line - t0, frames);
    if (g_probes_sent)
        fprintf(stderr, "probes: %d sent, %d answered, latency avg %.3fs max %.3fs\n",
                g_probes_sent, g_verdicts, g_verdicts ? g_lat_sum / g_verdicts : 0, g_lat_max);
    return 0;
}
