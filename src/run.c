#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

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

// One block-LG_MTP_N speculative step (shared by serve and -p): draft LG_MTP_N-1
// tokens (chained for the 2nd+), then verify the LG_MTP_N batch in one weight pass.
// Fills toks[LG_MTP_N] (toks[0]=best) and out[LG_MTP_N]; returns adv (1..LG_MTP_N) =
// the run of drafts that held, +1. Accumulates draft/verify time and draft/accept
// counts. An unavailable draft (no head / no chaining) is padded so the verify just
// rejects it. Greedy verify keeps the emitted text byte-identical to plain decode.
static int mtp_step(struct mtp *t, struct model *m, struct kvcache *kv, int best, int pos,
                    int *toks, int *out, double *t_draft, double *t_verify, int *n_draft, int *n_accept) {
    double s0 = now_sec();
    toks[0] = best;
    int d = mtp_draft(t, m, kv, best, pos);
    toks[1] = d >= 0 ? d : best;
    for (int j = 2; j < LG_MTP_N; j++) {
        int dj = mtp_draft_chain(t, m, kv, toks[j - 1], pos + j - 1);
        toks[j] = dj >= 0 ? dj : toks[j - 1];
    }
    double s1 = now_sec();
    int adv = model_forward_spec(m, kv, toks, pos, out);
    *t_draft += s1 - s0;
    *t_verify += now_sec() - s1;
    *n_draft += LG_MTP_N - 1;
    *n_accept += adv - 1;
    return adv;
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
#define SERVE_GEN 1024   // output cap per turn: greedy decode has no sampler and
                         // no repeat penalty, so a degenerate loop would otherwise
                         // spin until the context fills (observed: 8,098 tokens,
                         // 270 s, of the same sentence about a typo)

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

static void serve(const struct gguf_context *ctx, const char *path, const char *mmproj,
                  const char *syspath, const char *mtp_path) {
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
    if (md) model_prefill_reserve();   // size prefill buffers for a whole image span before any forward
    int eot = tokenizer_token_id(tk, "<turn|>");
    int eos = tokenizer_eos(tk);

    // The draft head (-mtp): block-2 speculative decode, same as generate().
    // A failed open falls back to plain greedy (NULL) — log either way so the
    // benchmark trace shows whether the spec path is actually live.
    struct mtp *t = mtp_path ? mtp_open(mtp_path, &m) : NULL;
    if (mtp_path)
        fprintf(stderr, "mtp: %s%s\n", t ? "speculative decode armed - " : "disabled, plain decode - ", mtp_path);

    // The system prefix (-sys): the skills/guidelines turn is prefilled ONCE
    // here and its cache rows saved; every session then starts at position
    // n_sys with the skills already in context, instead of paying their whole
    // prefill again per connection — the difference between an instant first
    // token and a many-second one when the skills run long (robotic chat
    // reconnects per exchange). BOS lives in this prefix when it exists.
    int n_sys = 0;
    if (syspath) {
        FILE *f = fopen(syspath, "rb");
        char *text = NULL;
        long fl = 0;
        if (f) { fseek(f, 0, SEEK_END); fl = ftell(f); fseek(f, 0, SEEK_SET); }
        if (!f || fl <= 0 || fl > 12000 || !(text = malloc((size_t)fl + 32)) ||
            fread(text, 1, (size_t)fl, f) != (size_t)fl) {
            fprintf(stderr, "-sys: cannot read %s (must be 1..12000 bytes)\n", syspath);
            if (f) fclose(f);
            free(text);
            return;
        }
        fclose(f);
        while (fl > 0 && (text[fl - 1] == '\n' || text[fl - 1] == '\r')) fl--;
        text[fl] = 0;
        size_t cap = (size_t)fl + 64;
        char *chat = malloc(cap);
        int *sysv = malloc(4096 * sizeof(int));
        if (!chat || !sysv) { free(text); free(chat); free(sysv); return; }
        snprintf(chat, cap, "<|turn>system\n%s<turn|>\n", text);
        int n = tokenizer_encode(tk, chat, sysv, 4096);
        free(text); free(chat);
        if (n <= 0 || n >= 4096 || n + 256 >= SERVE_SEQ) {
            fprintf(stderr, "-sys: prefix is %d tokens — too long for this server\n", n);
            free(sysv);
            return;
        }
        double t0 = now_sec();
        model_prefill(&m, &kv, sysv, n, 0);
        kvcache_save_prefix(&kv, n);
        free(sysv);
        n_sys = n;
        fprintf(stderr, "system prefix: %d tokens prefilled once in %.2fs (%s)\n",
                n_sys, now_sec() - t0, syspath);
    }

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
        // self-diagnosing on purpose: a relative path that LOOKS absolute is
        // the classic miss (PowerShell does not expand %TEMP% — that literal
        // directory really got created once), as is a directory created AT
        // the socket path after reading "the directory must exist"
        int err = sock_err();
        char full[576];
#ifdef _WIN32
        if (!_fullpath(full, path, sizeof full)) snprintf(full, sizeof full, "%s", path);
#else
        snprintf(full, sizeof full, "%s", path);
#endif
        struct stat st;
        fprintf(stderr, "bind/listen failed (error %d) on %s\n  resolved: %s\n  %s\n",
                err, path, full,
                stat(full, &st) == 0 && (st.st_mode & S_IFDIR)
                ? "a DIRECTORY sits at the socket path itself - remove it; only the PARENT must exist"
                : "the socket's PARENT directory must exist, and the path itself must be free");
        return;
    }
    fprintf(stderr, "listening on %s - one conversation per connection, Ctrl-C to stop\n", path);

    for (;;) {
        sock_t c = accept(ls, NULL, NULL);
        if (c == INVALID_SOCKET) continue;
        fprintf(stderr, "session start\n");
        int pos = n_sys;                                 // each session restarts right after the
        kvcache_restore_prefix(&kv, n_sys);              // system prefix (at 0 when there is none);
                                                         // restore repairs ring rows a long previous
                                                         // session may have wrapped over
        char line[8192], chat[8704];
        int promptv[4096];
        for (;;) {
            // A turn: zero or more typed frames (see media.h) — media spans
            // and 'T' text that lands between them (a video is "00:00 " frame
            // " 00:01 " frame ...) — then the text line. The first byte tells
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
                double e0 = now_sec();
                float *rows = hdr[0] == MEDIA_FRAME_IMAGE && len == 3u * w * h
                            ? media_embed_image(md, (uint8_t *)payload, w, h, &seg[n_seg].n)
                            : hdr[0] == MEDIA_FRAME_AUDIO
                            ? media_embed_audio(md, (int16_t *)payload, (int)(len / 2), &seg[n_seg].n)
                            : NULL;
                free(payload);
                if (!rows) { dead = 1; break; }
                seg[n_seg].rows = rows;
                in_media += seg[n_seg].n;
                fprintf(stderr, "media: %c frame -> %d tokens in %.1fs\n", hdr[0], seg[n_seg].n, now_sec() - e0);
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
            // The turn is assembled as alternating text and embedding spans —
            // text accumulates (turn opener, 'T' frames, media markers) and is
            // encoded right before each media span — but the whole turn
            // prefills as ONE mixed stream: ids >= 0 are text tokens, ids < 0
            // index the media rows gathered alongside. Prefilled separately,
            // every text/media seam padded its own chunk to a full weight
            // pass; a short camera turn spent a third of its prefill on the
            // seams. Only the very first encode of the conversation keeps its
            // BOS (when a -sys prefix exists, the BOS already lives there);
            // the first turn opens plainly, later turns first close the
            // previous model turn, whose <turn|> was never fed back.
            int skip = (pos == 0) ? 0 : 1;               // tokenizer_encode always prepends BOS
            int n = 0, total = 0, best;
            int idcap = in_media + in_text + (int)strlen(line) + 16 * n_seg + 192;
            int *ids = malloc((size_t)idcap * sizeof *ids);
            float *mrows = in_media ? malloc((size_t)in_media * m.cfg.n_embd * 4) : NULL;
            if (!ids || (in_media && !mrows)) {
                free(ids); free(mrows);
                for (int i = 0; i < n_seg; i++) { free(seg[i].rows); free(seg[i].text); }
                fprintf(stderr, "turn buffer: out of memory\n");
                break;
            }
            int nid = 0, nmr = 0;
            int tl = snprintf(chat, sizeof chat, pos == n_sys ? "<|turn>user\n" : "<turn|>\n<|turn>user\n");
            for (int i = 0; i < n_seg; i++) {
                if (seg[i].kind == MEDIA_FRAME_TEXT) {
                    tl += snprintf(chat + tl, sizeof chat - tl, "%s", seg[i].text);
                    free(seg[i].text);
                    continue;
                }
                tl += snprintf(chat + tl, sizeof chat - tl, "%s",
                               seg[i].kind == MEDIA_FRAME_IMAGE ? MEDIA_IMG_BEG : MEDIA_AUD_BEG);
                n = tokenizer_encode(tk, chat, promptv, 4096);
                for (int j = skip; j < n; j++) ids[nid++] = promptv[j];
                skip = 1;
                memcpy(mrows + (size_t)nmr * m.cfg.n_embd, seg[i].rows, (size_t)seg[i].n * m.cfg.n_embd * 4);
                for (int r = 0; r < seg[i].n; r++) ids[nid++] = -(nmr + r) - 1;
                nmr += seg[i].n;
                free(seg[i].rows);
                tl = snprintf(chat, sizeof chat, "%s",
                              seg[i].kind == MEDIA_FRAME_IMAGE ? MEDIA_IMG_END : MEDIA_AUD_END);
            }
            snprintf(chat + tl, sizeof chat - tl, "%s<turn|>\n<|turn>model\n", line);
            n = tokenizer_encode(tk, chat, promptv, 4096);
            for (int j = skip; j < n - 1; j++) ids[nid++] = promptv[j];
            model_prefill_mixed(&m, &kv, mrows, ids, nid, pos);
            pos += nid;
            total = nid + 1;                             // + the head forward below
            free(ids); free(mrows);
            best = model_forward_next(&m, &kv, promptv[n - 1], pos++);
            double t1 = now_sec();                       // prefill done (incl. the first pick)
            int g = 0, fail = 0;                          // g = tokens streamed this turn
            double t_draft = 0, t_verify = 0;
            int n_draft = 0, n_accept = 0;
            for (;;) {                                   // stream raw token text, turn end included
                if (client_reset(c) || send_piece(c, tokenizer_token_text(tk, best)) != 0) { fail = 1; break; }
                g++;
                if (best == eot || best == eos || pos + 1 >= SERVE_SEQ) break;
                if (g >= SERVE_GEN) {                    // runaway turn: end it ourselves
                    send_piece(c, " [SERVE_GEN cap]<turn|>");
                    fprintf(stderr, "turn capped at %d tokens without an end-of-turn\n", SERVE_GEN);
                    break;
                }
                if (!t || pos + LG_MTP_N - 1 >= SERVE_SEQ) {   // plain decode: no head, or no room for a full spec block
                    best = model_forward_next(&m, &kv, best, pos++);
                    continue;
                }
                // block-LG_MTP_N speculation: draft LG_MTP_N-1 tokens (chained), verify
                // the batch in one weight pass — up to LG_MTP_N tokens for one pass's
                // weight traffic. Greedy verify keeps the streamed text byte-identical.
                int toks[LG_MTP_N], out[LG_MTP_N];
                int adv = mtp_step(t, &m, &kv, best, pos, toks, out, &t_draft, &t_verify, &n_draft, &n_accept);
                pos += adv;
                int brk = 0;
                for (int e = 1; e < adv && !brk; e++) {  // stream the confirmed drafts toks[1..adv-1]
                    if (client_reset(c) || send_piece(c, tokenizer_token_text(tk, toks[e])) != 0) { fail = 1; brk = 1; break; }
                    g++;
                    if (toks[e] == eot || toks[e] == eos || pos + 1 >= SERVE_SEQ) { brk = 1; break; }
                    if (g >= SERVE_GEN) {
                        send_piece(c, " [SERVE_GEN cap]<turn|>");
                        fprintf(stderr, "turn capped at %d tokens without an end-of-turn\n", SERVE_GEN);
                        brk = 1; break;
                    }
                }
                if (brk) break;
                best = out[adv - 1];
            }
            double dt = now_sec() - t1, dp = t1 - t0;
            fprintf(stderr, "turn: %d in %.2fs (%.1f tok/s), %d out %.2fs (%.1f tok/s)\n",
                    total, dp, total / (dp > 0 ? dp : 1e-9),
                    g, dt, g / (dt > 0 ? dt : 1e-9));
            if (t && n_draft)
                fprintf(stderr, "mtp:    accepted %d/%d drafts (%.1f%%) — %d rounds: draft %.2fs (%.1fms ea), verify %.2fs (%.1fms ea)\n",
                        n_accept, n_draft, 100.0 * n_accept / n_draft,
                        n_draft, t_draft, 1e3 * t_draft / n_draft, t_verify, 1e3 * t_verify / n_draft);
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

// Print one generated token unless it is scaffolding; tracks the thought-
// channel-name state (the name between <|channel> and <channel|> stays hidden).
static void emit_token(struct tokenizer *tk, int tok, int *in_thought, int ch_open, int ch_close) {
    if (tok == ch_open)  { *in_thought = 1; return; }
    if (tok == ch_close) { *in_thought = 0; return; }
    if (!*in_thought && !tokenizer_is_special(tk, tok)) {
        print_piece(tokenizer_token_text(tk, tok));
        fflush(stdout);
    }
}

// Tokenize the prompt under the Gemma chat template, generate greedily, report tok/s.
// `mtp_path` (optional) loads a gemma4-assistant draft head and decodes
// SPECULATIVELY at block 2: draft the successor of each fresh token, then
// verify [token, draft] as one batched target step — two tokens for one
// pass's weight traffic when the draft holds. Greedy verification means the
// output is byte-identical to plain greedy decoding, always; only tok/s and
// the acceptance rate move.
static void generate(const struct gguf_context *ctx, const char *prompt, const char *mtp_path) {
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

    struct mtp *t = mtp_path ? mtp_open(mtp_path, &m) : NULL;
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

    // greedy generation — plain, or block-LG_MTP_N speculative with -mtp. g counts
    // EMITTED tokens, so the spec path stops at exactly the same N_GEN boundary as
    // plain decoding (an accepted draft past the cap is simply never printed).
    double tg = now_sec();
    double t_draft = 0, t_verify = 0;
    int g = 0, n_draft = 0, n_accept = 0;
    while (g < N_GEN) {
        if (best == eot || best == eos) break;               // end of turn
        emit_token(tk, best, &in_thought, ch_open, ch_close);
        g++;
        if (!t || g >= N_GEN || pos + LG_MTP_N - 1 >= kv.max_seq) {
            if (g < N_GEN) best = model_forward_next(&m, &kv, best, pos++);
            continue;
        }
        int toks[LG_MTP_N], out[LG_MTP_N];
        int adv = mtp_step(t, &m, &kv, best, pos, toks, out, &t_draft, &t_verify, &n_draft, &n_accept);
        pos += adv;
        int stop = 0;
        for (int e = 1; e < adv && !stop; e++) {             // emit confirmed drafts toks[1..adv-1]
            if (g >= N_GEN) { stop = 1; break; }
            emit_token(tk, toks[e], &in_thought, ch_open, ch_close);
            g++;
            if (toks[e] == eot || toks[e] == eos) { stop = 1; break; }
        }
        if (stop) break;
        best = out[adv - 1];
    }
    double t_gen = now_sec() - tg;

    printf("\n\n");
    fprintf(stderr, "prompt: %d tokens in %.2fs (%.2f tok/s)\n",
            n_prompt, t_prompt, n_prompt / (t_prompt > 0 ? t_prompt : 1e-9));
    fprintf(stderr, "gen:    %d tokens in %.2fs (%.2f tok/s)\n",
            g, t_gen, g / (t_gen > 0 ? t_gen : 1e-9));
    if (t && n_draft)
        fprintf(stderr, "mtp:    accepted %d/%d drafts (%.1f%%) — %d rounds: draft %.2fs (%.1fms ea), verify %.2fs (%.1fms ea)\n",
                n_accept, n_draft, 100.0 * n_accept / n_draft,
                n_draft, t_draft, 1e3 * t_draft / n_draft, t_verify, 1e3 * t_verify / n_draft);

    mtp_free(t);
    kvcache_free(&kv);
    model_free(&m);
    tokenizer_free(tk);
}

int main(int argc, char **argv) {
    const char *model = NULL, *prompt = NULL, *spath = NULL, *cpath = NULL, *mmproj = NULL, *mtp = NULL, *syspath = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc)       model  = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc)  prompt = argv[++i];
        else if (!strcmp(argv[i], "-s") && i + 1 < argc)  spath  = argv[++i];
        else if (!strcmp(argv[i], "-c") && i + 1 < argc)  cpath  = argv[++i];
        else if (!strcmp(argv[i], "-mm") && i + 1 < argc) mmproj = argv[++i];
        else if (!strcmp(argv[i], "-mtp") && i + 1 < argc) mtp   = argv[++i];
        else if (!strcmp(argv[i], "-sys") && i + 1 < argc) syspath = argv[++i];
    }
    if (cpath) { client(cpath); return 0; }              // client mode needs no model
    if (!model) {
        printf("Usage: %s -m <model.gguf> [-mm <mmproj.gguf>] [-mtp <assistant.gguf>] [-sys <skills.txt>] [-p \"prompt\" | -s <socket>]\n"
               "       %s -c <socket>\n", argv[0], argv[0]);
        return 1;
    }

    struct gguf_context *ctx = load_gguf(model);
    if (!ctx) return 1;

    gguf_dump(ctx);

    struct config cfg;
    if (config_load(&cfg, ctx) == 0) { printf("\n"); config_print(&cfg); }

    if (spath)       serve(ctx, spath, mmproj, syspath, mtp);
    else if (prompt) { printf("\n"); generate(ctx, prompt, mtp); }

    free_gguf(ctx);
    return 0;
}
