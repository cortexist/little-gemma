/* ttfc_dictate.c - native Windows port of ttfc_dictate.py (Windows CPython has
 * no AF_UNIX, so the paced dictation client must be native).  Semantics mirror
 * the python client exactly so A5000 numbers are comparable with the Orin's:
 *   ttfc_dictate.exe <sock> <line_file> <words_per_frame> <frames_per_sec>
 * wpf=0 sends the whole line at once; otherwise words go as typed 'T' frames
 * (MAGIC 0x01, u16 w=0, u16 h=0, u32 len), one sleep(1/fps) after EVERY frame
 * (the python client sleeps after the last frame too, before the "\n").
 * Clocks (all relative to the "\n" send): ttft = first reply byte, ttfc =
 * first clause boundary [,;:.!?] followed by space-like/EOB, ttfs = first
 * sentence boundary [.!?].  Boundary scan runs on the "speakable" text:
 * <|channel>..<channel|> thought spans and <..> tags (1..24 chars) removed. */
#include <winsock2.h>
#include <afunix.h>
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "winmm.lib")

#define BUFMAX (1 << 20)

static double now(void) {
    static LARGE_INTEGER f;
    LARGE_INTEGER c;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / f.QuadPart;
}

/* python: re.sub(r"<\|channel>.*?(<channel\|>|$)", "", s, flags=re.S) then
 * re.sub(r"<[^<>]{1,24}>", "", s) */
static void speakable(const char *raw, char *out) {
    static char tmp[BUFMAX];
    const char *p = raw;
    char *o = tmp;
    while (*p) {
        if (!strncmp(p, "<|channel>", 10)) {
            const char *e = strstr(p + 10, "<channel|>");
            p = e ? e + 10 : p + strlen(p);
            continue;
        }
        *o++ = *p++;
    }
    *o = 0;
    p = tmp; o = out;
    while (*p) {
        if (*p == '<') {
            const char *q = p + 1;
            while (*q && *q != '<' && *q != '>' && q - p <= 24) q++;
            if (*q == '>' && q - p >= 2) { p = q + 1; continue; }
        }
        *o++ = *p++;
    }
    *o = 0;
}

static int is_after(char c) {      /* the lookahead class [\s\*\)\"'\]] */
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v' ||
           c == '*' || c == ')' || c == '"' || c == '\'' || c == ']';
}

/* first index of a char in `set` whose successor is space-like or end-of-buffer;
 * -1 if none (end-of-buffer counts: the stream is re-scanned as bytes arrive,
 * matching the python regex's `|$`). */
static int boundary(const char *s, const char *set) {
    for (int i = 0; s[i]; i++)
        if (strchr(set, s[i]) && (s[i + 1] == 0 || is_after(s[i + 1])))
            return i;
    return -1;
}

int main(int argc, char **argv) {
    if (argc != 5) { fprintf(stderr, "usage: %s <sock> <line_file> <wpf> <fps>\n", argv[0]); return 1; }
    int wpf = atoi(argv[3]);
    double fps = atof(argv[4]);

    FILE *f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    static char line[BUFMAX];
    size_t n = fread(line, 1, sizeof line - 1, f);
    fclose(f);
    line[n] = 0;
    while (n && (line[n - 1] == '\n' || line[n - 1] == '\r' || line[n - 1] == ' ')) line[--n] = 0;

    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
    timeBeginPeriod(1);
    SOCKET s = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un sa;
    memset(&sa, 0, sizeof sa);
    sa.sun_family = AF_UNIX;
    strcpy(sa.sun_path, argv[1]);
    if (connect(s, (struct sockaddr *)&sa, sizeof sa)) { fprintf(stderr, "connect failed\n"); return 1; }

    double t0 = now();
    if (wpf == 0) {
        send(s, line, (int)strlen(line), 0);
        send(s, "\n", 1, 0);
    } else {
        char *w = line, frame[8192];
        while (*w) {
            char *g = w;                              /* group of wpf words */
            for (int k = 0; k < wpf && *w; k++) {
                while (*w && *w != ' ') w++;
                if (*w && k + 1 < wpf) w++;
            }
            int glen = (int)(w - g), off = 0;
            if (g != line) { frame[10] = ' '; off = 1; } /* leading space on groups after the first */
            frame[0] = 0x01; frame[1] = 'T';             /* MAGIC 'T' + u16 w,h + u32 len = 10-byte header */
            unsigned short z = 0; unsigned int len = (unsigned)(glen + off);
            memcpy(frame + 2, &z, 2); memcpy(frame + 4, &z, 2); memcpy(frame + 6, &len, 4);
            memcpy(frame + 10 + off, g, (size_t)glen);
            send(s, frame, 10 + glen + off, 0);
            if (fps > 0) Sleep((DWORD)(1000.0 / fps));
            while (*w == ' ') w++;
        }
        send(s, "\n", 1, 0);
    }
    double t_line = now();

    static char buf[BUFMAX], sp[BUFMAX];
    int bn = 0;
    double t_first = 0, t_clause = 0, t_sent = 0;
    char clause[4096] = "";
    while (!strstr(buf, "<turn|>")) {
        int k = recv(s, buf + bn, 4096, 0);
        if (k <= 0) break;
        if (!t_first) t_first = now();
        bn += k; buf[bn] = 0;
        speakable(buf, sp);
        if (!t_clause) {
            int i = boundary(sp, ",;:.!?");
            if (i >= 0) {
                t_clause = now();
                int b = 0;
                while (sp[b] == ' ' || sp[b] == '\n') b++;
                int e = i + 1;
                int m = e - b < (int)sizeof clause - 1 ? e - b : (int)sizeof clause - 1;
                memcpy(clause, sp + b, (size_t)m); clause[m] = 0;
            }
        }
        if (!t_sent && boundary(sp, ".!?") >= 0) t_sent = now();
    }
    closesocket(s);

    printf("ttft %.3fs | ttfc %.3fs | ttfs %.3fs | send window %.1fs\n",
           t_first ? t_first - t_line : -1, t_clause ? t_clause - t_line : -1,
           t_sent ? t_sent - t_line : -1, t_line - t0);
    printf("first clause: '%s'\n", clause);
    for (char *p = sp; *p; p++) if (*p == '\n') *p = ' ';
    printf("reply: %.220s\n", sp);
    return 0;
}
