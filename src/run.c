#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <winsock2.h>   // must precede windows.h
#include <windows.h>
#include <afunix.h>     // AF_UNIX on Windows 10 1803+
typedef SOCKET sock_t;
#define sock_close closesocket
static int sock_init(void) { WSADATA w; return WSAStartup(MAKEWORD(2, 2), &w); }
static int sock_err(void) { return WSAGetLastError(); }
#else
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
typedef int sock_t;
#define INVALID_SOCKET (-1)
#define sock_close close
// A dead client must surface as a send() error, not a fatal signal.
static int sock_init(void) { signal(SIGPIPE, SIG_IGN); return 0; }
static int sock_err(void) { return errno; }
#endif

#include "gguf.h"
#include "model.h"
#include "tokenizer.h"
#include "media.h"

#define N_GEN 256  // max tokens to generate (stops early on EOS)

// Wall-clock seconds. clock() is WALL time on Windows but CPU time on POSIX —
// with the GPU doing the work the CPU mostly idle-waits, so the Orin reported
// a glorious 3,400 tok/s before this was caught.
static double now_sec(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

// Print a token's text with the ▁ marker (U+2581 = e2 96 81) rendered as a space.
static void print_piece(const char *s) {
    if (!s) return;
    for (const char *p = s; *p; p++) {
        if ((unsigned char)p[0] == 0xE2 && (unsigned char)p[1] == 0x96 && (unsigned char)p[2] == 0x81)
            { putchar(' '); p += 2; } else putchar(*p);
    }
}

// ---- socket serving (-s) ---------------------------------------------------
//
// A connection IS a conversation: each newline-terminated line from the client
// is a user turn, and the model's turn streams back as RAW token text — special
// tokens included, closed by the literal "<turn|>" — so downstream tools split
// turns and handle channels themselves. The kv cache lives for the connection,
// which makes multi-turn context free; close the connection to end the session.
// stdout/stderr are logging only. Transport concerns (TCP, TLS, auth) belong to
// socat and friends. Cancellation is cooperative: a dead client makes send()
// fail, and the generation loop stops at the next token — one forward (~5-17ms)
// is the unit of work that cannot be interrupted, and never needs to be.

#define SERVE_SEQ 8192   // context budget per conversation

static int send_all(sock_t s, const char *buf, int n) {
    while (n > 0) {
        int k = (int)send(s, buf, n, 0);
        if (k <= 0) return -1;
        buf += k; n -= k;
    }
    return 0;
}

// One token's text onto the wire, with the ▁ marker rendered as a space.
static int send_piece(sock_t s, const char *t) {
    char buf[512]; int n = 0;
    if (!t) return 0;
    for (const char *p = t; *p && n < (int)sizeof buf - 1; p++) {
        if ((unsigned char)p[0] == 0xE2 && (unsigned char)p[1] == 0x96 && (unsigned char)p[2] == 0x81)
            { buf[n++] = ' '; p += 2; } else buf[n++] = *p;
    }
    return send_all(s, buf, n);
}

// Did the connection break? A buffered send() succeeds whether or not anyone
// reads it, so failure is only visible to recv: peek non-blocking, without
// consuming, so a healthy client's pipelined next turn stays intact. A clean
// half-close (recv 0 — e.g. socat after stdin EOF, still reading the answer)
// is NOT a break: the current turn finishes and the session ends at the next
// read. Only a hard error aborts mid-turn.
static int client_reset(sock_t s) {
    char c;
#ifdef _WIN32
    u_long nb = 1; ioctlsocket(s, FIONBIO, &nb);
    int k = (int)recv(s, &c, 1, MSG_PEEK);
    int err = WSAGetLastError();
    nb = 0; ioctlsocket(s, FIONBIO, &nb);
    return k < 0 && err != WSAEWOULDBLOCK;
#else
    int k = (int)recv(s, &c, 1, MSG_PEEK | MSG_DONTWAIT);
    return k < 0 && errno != EAGAIN && errno != EWOULDBLOCK;
#endif
}

// One newline-terminated user turn (CR stripped), whose first byte `first`
// was already consumed by the frame/text dispatch. -1 on close or error.
static int recv_line(sock_t s, char *buf, int cap, char first) {
    int n = 0;
    char c = first;
    for (;;) {
        if (c == '\n') break;
        if (c != '\r' && n < cap - 1) buf[n++] = c;
        int k = (int)recv(s, &c, 1, 0);
        if (k <= 0) return -1;
    }
    buf[n] = 0;
    return n;
}

// Read exactly n bytes (a frame header or payload). -1 on close or error.
static int recv_n(sock_t s, void *buf, int n) {
    char *p = (char *)buf;
    while (n > 0) {
        int k = (int)recv(s, p, n, 0);
        if (k <= 0) return -1;
        p += k; n -= k;
    }
    return 0;
}

static void serve(const struct gguf_context *ctx, const char *path, const char *mmproj) {
    struct tokenizer *tk = tokenizer_init(ctx);
    if (!tk) { fprintf(stderr, "tokenizer init failed\n"); return; }
    struct model m;
    if (model_init(&m, ctx) != 0) { tokenizer_free(tk); return; }
    struct kvcache kv;
    if (kvcache_init(&kv, &m, SERVE_SEQ) != 0) { model_free(&m); tokenizer_free(tk); return; }
    // the multimodal projector (-mm): without it, media frames end the session
    struct media *md = NULL;
    if (mmproj && !(md = media_open(mmproj, m.cfg.n_embd))) {
        model_free(&m); tokenizer_free(tk); return;
    }
    int eot = tokenizer_token_id(tk, "<turn|>");
    int eos = tokenizer_eos(tk);

    sock_t ls = INVALID_SOCKET;
    struct sockaddr_un sa;
    memset(&sa, 0, sizeof sa);
    sa.sun_family = AF_UNIX;
    if (sock_init() != 0 || strlen(path) >= sizeof sa.sun_path ||
        (ls = socket(AF_UNIX, SOCK_STREAM, 0)) == INVALID_SOCKET) {
        fprintf(stderr, "socket setup failed (path too long?)\n"); return;
    }
    strcpy(sa.sun_path, path);
    remove(path);                                       // stale socket from a previous run
    if (bind(ls, (struct sockaddr *)&sa, sizeof sa) != 0 || listen(ls, 4) != 0) {
        fprintf(stderr, "bind/listen on %s failed (error %d) - note the socket's directory must exist\n",
                path, sock_err());
        return;
    }
    fprintf(stderr, "listening on %s - one conversation per connection, Ctrl-C to stop\n", path);

    for (;;) {
        sock_t c = accept(ls, NULL, NULL);
        if (c == INVALID_SOCKET) continue;
        fprintf(stderr, "session start\n");
        int pos = 0;                                     // kv cache restarts with each session
        char line[8192], chat[8704];
        int promptv[4096];
        for (;;) {
            // A turn: zero or more typed frames (see media.h) — media spans
            // and 'T' text that lands between them (a video is "0:01" frame
            // "0:02" frame ...) — then the text line. The first byte tells
            // frame from line, so a text-only client speaks the same protocol
            // it always did.
            #define MAX_SEG 32
            struct { char kind; float *rows; int n; char *text; } seg[MAX_SEG];
            int n_seg = 0, in_media = 0, in_text = 0;
            char c0 = 0;
            int dead = 0;
            for (;;) {
                if (recv(c, &c0, 1, 0) != 1) { dead = 1; break; }
                if ((unsigned char)c0 != MEDIA_FRAME_MAGIC) break;
                unsigned char hdr[MEDIA_FRAME_HDR - 1];
                uint16_t w, h; uint32_t len;
                if (recv_n(c, hdr, sizeof hdr) < 0) { dead = 1; break; }
                memcpy(&w, hdr + 1, 2); memcpy(&h, hdr + 3, 2); memcpy(&len, hdr + 5, 4);
                char *payload = len && len <= (32u << 20) ? malloc((size_t)len + 1) : NULL;
                if (!payload || recv_n(c, payload, (int)len) < 0) { free(payload); dead = 1; break; }
                if (n_seg == MAX_SEG) { fprintf(stderr, "too many frames in one turn\n"); free(payload); dead = 1; break; }
                seg[n_seg].kind = (char)hdr[0];
                seg[n_seg].rows = NULL;
                seg[n_seg].text = NULL;
                if (hdr[0] == MEDIA_FRAME_TEXT && len <= 4096) {
                    payload[len] = 0;
                    seg[n_seg].text = payload;
                    in_text += (int)len;
                    n_seg++;
                    continue;
                }
                if (!md) { fprintf(stderr, "media frame but no -mm projector\n"); free(payload); dead = 1; break; }
                float *rows = hdr[0] == MEDIA_FRAME_IMAGE && len == 3u * w * h
                            ? media_embed_image(md, (uint8_t *)payload, w, h, &seg[n_seg].n)
                            : hdr[0] == MEDIA_FRAME_AUDIO
                            ? media_embed_audio(md, (int16_t *)payload, (int)(len / 2), &seg[n_seg].n)
                            : NULL;
                free(payload);
                if (!rows) { dead = 1; break; }
                seg[n_seg].rows = rows;
                in_media += seg[n_seg].n;
                fprintf(stderr, "media: %c frame -> %d tokens\n", hdr[0], seg[n_seg].n);
                n_seg++;
            }
            if (dead || recv_line(c, line, sizeof line, c0) < 0) {
                for (int i = 0; i < n_seg; i++) { free(seg[i].rows); free(seg[i].text); }
                break;
            }
            // budget: chars bound tokens from above, so this never under-reserves;
            // the chat buffer must also hold every 'T' chunk plus the line
            if (pos + in_media + in_text + (int)strlen(line) + 64 >= SERVE_SEQ ||
                in_text + (int)strlen(line) + 256 >= (int)sizeof chat) {
                for (int i = 0; i < n_seg; i++) { free(seg[i].rows); free(seg[i].text); }
                fprintf(stderr, "context full\n");
                break;
            }
            double t0 = now_sec();
            // The turn is assembled as alternating text and embedding spans:
            // text accumulates (turn opener, 'T' frames, media markers) and is
            // encoded+prefilled in one go right before each media span, whose
            // rows then prefill as embeddings; the question line closes the
            // turn. Only the very first encode of the conversation keeps its
            // BOS; the first turn opens it, later turns first close the
            // previous model turn, whose <turn|> was never fed back.
            int skip = pos == 0 ? 0 : 1;                 // tokenizer_encode always prepends BOS
            int n = 0, total = 0, best;
            int tl = snprintf(chat, sizeof chat, pos == 0 ? "<|turn>user\n" : "<turn|>\n<|turn>user\n");
            for (int i = 0; i < n_seg; i++) {
                if (seg[i].kind == MEDIA_FRAME_TEXT) {
                    tl += snprintf(chat + tl, sizeof chat - tl, "%s", seg[i].text);
                    free(seg[i].text);
                    continue;
                }
                tl += snprintf(chat + tl, sizeof chat - tl, "%s",
                               seg[i].kind == MEDIA_FRAME_IMAGE ? MEDIA_IMG_BEG : MEDIA_AUD_BEG);
                n = tokenizer_encode(tk, chat, promptv, 4096);
                model_prefill(&m, &kv, promptv + skip, n - skip, pos);
                pos += n - skip;
                total += n - skip;
                skip = 1;
                model_prefill_embd(&m, &kv, seg[i].rows, seg[i].n, pos);
                pos += seg[i].n;
                total += seg[i].n;
                free(seg[i].rows);
                tl = snprintf(chat, sizeof chat, "%s",
                              seg[i].kind == MEDIA_FRAME_IMAGE ? MEDIA_IMG_END : MEDIA_AUD_END);
            }
            snprintf(chat + tl, sizeof chat - tl, "%s<turn|>\n<|turn>model\n", line);
            n = tokenizer_encode(tk, chat, promptv, 4096);
            model_prefill(&m, &kv, promptv + skip, n - 1 - skip, pos);
            pos += n - 1 - skip;
            total += n - skip;
            best = model_forward_next(&m, &kv, promptv[n - 1], pos++);
            double t1 = now_sec();                       // prefill done (incl. the first pick)
            int g = 0, fail = 0;
            for (;; g++) {                               // stream raw token text, turn end included
                if (client_reset(c) || send_piece(c, tokenizer_token_text(tk, best)) != 0) { fail = 1; break; }
                if (best == eot || best == eos || pos + 1 >= SERVE_SEQ) break;
                best = model_forward_next(&m, &kv, best, pos++);
            }
            double dt = now_sec() - t1, dp = t1 - t0;
            fprintf(stderr, "turn: %d in %.2fs (%.1f tok/s), %d out %.2fs (%.1f tok/s)\n",
                    total, dp, total / (dp > 0 ? dp : 1e-9),
                    g + 1, dt, (g + 1) / (dt > 0 ? dt : 1e-9));
            if (fail || pos + 1 >= SERVE_SEQ) break;     // client gone or context full
        }
        sock_close(c);
        fprintf(stderr, "session end\n");
    }
}

// ---- minimal client (-c) ----------------------------------------------------
// For systems without socat (native Windows): pump stdin lines to the server
// and copy each streamed model turn to stdout, reading until its "<turn|>".
static void client(const char *path) {
    struct sockaddr_un sa;
    memset(&sa, 0, sizeof sa);
    sa.sun_family = AF_UNIX;
    sock_t s;
    if (sock_init() != 0 || strlen(path) >= sizeof sa.sun_path ||
        (s = socket(AF_UNIX, SOCK_STREAM, 0)) == INVALID_SOCKET) {
        fprintf(stderr, "socket setup failed (path too long?)\n"); return;
    }
    strcpy(sa.sun_path, path);
    if (connect(s, (struct sockaddr *)&sa, sizeof sa) != 0) {
        fprintf(stderr, "connect to %s failed\n", path); return;
    }
    char line[8192];
    while (fgets(line, sizeof line, stdin)) {
        size_t len = strlen(line);
        if (len + 1 < sizeof line && (len == 0 || line[len - 1] != '\n')) { line[len++] = '\n'; line[len] = 0; }
        if (send_all(s, line, (int)len) != 0) break;
        char hist[4096 + 8];                             // 7-byte overlap catches a split "<turn|>"
        int hn = 0, done = 0;
        while (!done) {
            int k = (int)recv(s, hist + hn, 4096, 0);
            if (k <= 0) { sock_close(s); return; }
            fwrite(hist + hn, 1, (size_t)k, stdout);
            fflush(stdout);
            hist[hn + k] = 0;
            if (strstr(hist, "<turn|>")) done = 1;
            int keep = hn + k < 7 ? hn + k : 7;
            memmove(hist, hist + hn + k - keep, (size_t)keep);
            hn = keep;
        }
        putchar('\n');
        fflush(stdout);
    }
    sock_close(s);
}

// Tokenize the prompt under the Gemma chat template, generate greedily, report tok/s.
static void generate(const struct gguf_context *ctx, const char *prompt) {
    struct tokenizer *tk = tokenizer_init(ctx);
    if (!tk) { fprintf(stderr, "tokenizer init failed\n"); return; }

    // Wrap the user prompt in gemma4's chat format so instruction-tuned models
    // respond properly. <|turn> and <turn|> tokenize as atomic special tokens;
    // BOS is added by tokenizer_encode.
    size_t cap = strlen(prompt) + 64;
    char *chat = malloc(cap);
    snprintf(chat, cap, "<|turn>user\n%s<turn|>\n<|turn>model\n", prompt);

    int promptv[4096];
    int n_prompt = tokenizer_encode(tk, chat, promptv, 4096);
    free(chat);

    struct model m;
    if (model_init(&m, ctx) != 0) { tokenizer_free(tk); return; }
    struct kvcache kv;
    if (kvcache_init(&kv, &m, n_prompt + N_GEN + 1) != 0) {
        model_free(&m); tokenizer_free(tk); return;
    }
    int pos = 0;
    // Stop at end-of-turn. gemma4's turn end is <turn|>; the gguf eos_token_id is
    // <turn|> on E2B but <eos> on 12B, so stop on either.
    int eot = tokenizer_token_id(tk, "<turn|>");
    int eos = tokenizer_eos(tk);
    // Thinking models name a channel between <|channel> and <channel|> (e.g.
    // "thought"); hide that channel name so only the content prints.
    int ch_open = tokenizer_token_id(tk, "<|channel>");
    int ch_close = tokenizer_token_id(tk, "<channel|>");
    int in_thought = 0;

    // echo the user's prompt (not the chat scaffolding)
    printf("%s\n", prompt);
    fflush(stdout);

    // prefill: the last prompt token's forward also picks the first generated token
    double tp = now_sec();
    model_prefill(&m, &kv, promptv, n_prompt - 1, 0);
    pos = n_prompt - 1;
    int best = model_forward_next(&m, &kv, promptv[n_prompt - 1], pos++);
    double t_prompt = now_sec() - tp;

    // greedy generation
    double tg = now_sec();
    int g = 0;
    for (; g < N_GEN; g++) {
        if (best == eot || best == eos) break;               // end of turn
        if (best == ch_open)  in_thought = 1;                // channel name starts
        else if (best == ch_close) in_thought = 0;           // channel name ends
        else if (!in_thought && !tokenizer_is_special(tk, best)) {
            print_piece(tokenizer_token_text(tk, best));
            fflush(stdout);
        }
        best = model_forward_next(&m, &kv, best, pos++);
    }
    double t_gen = now_sec() - tg;

    printf("\n\n");
    fprintf(stderr, "prompt: %d tokens in %.2fs (%.2f tok/s)\n",
            n_prompt, t_prompt, n_prompt / (t_prompt > 0 ? t_prompt : 1e-9));
    fprintf(stderr, "gen:    %d tokens in %.2fs (%.2f tok/s)\n",
            g, t_gen, g / (t_gen > 0 ? t_gen : 1e-9));

    kvcache_free(&kv);
    model_free(&m);
    tokenizer_free(tk);
}

int main(int argc, char **argv) {
    const char *model = NULL, *prompt = NULL, *spath = NULL, *cpath = NULL, *mmproj = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc)       model  = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc)  prompt = argv[++i];
        else if (!strcmp(argv[i], "-s") && i + 1 < argc)  spath  = argv[++i];
        else if (!strcmp(argv[i], "-c") && i + 1 < argc)  cpath  = argv[++i];
        else if (!strcmp(argv[i], "-mm") && i + 1 < argc) mmproj = argv[++i];
    }
    if (cpath) { client(cpath); return 0; }              // client mode needs no model
    if (!model) {
        printf("Usage: %s -m <model.gguf> [-mm <mmproj.gguf>] [-p \"prompt\" | -s <socket>]\n"
               "       %s -c <socket>\n", argv[0], argv[0]);
        return 1;
    }

    struct gguf_context *ctx = load_gguf(model);
    if (!ctx) return 1;

    gguf_dump(ctx);

    struct config cfg;
    if (config_load(&cfg, ctx) == 0) { printf("\n"); config_print(&cfg); }

    if (spath)       serve(ctx, spath, mmproj);
    else if (prompt) { printf("\n"); generate(ctx, prompt); }

    free_gguf(ctx);
    return 0;
}
