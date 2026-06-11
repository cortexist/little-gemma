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

#define N_GEN 256  // max tokens to generate (stops early on EOS)

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

// One newline-terminated user turn (CR stripped). -1 on close or error.
static int recv_line(sock_t s, char *buf, int cap) {
    int n = 0;
    for (;;) {
        char c;
        int k = (int)recv(s, &c, 1, 0);
        if (k <= 0) return -1;
        if (c == '\n') break;
        if (c != '\r' && n < cap - 1) buf[n++] = c;
    }
    buf[n] = 0;
    return n;
}

static void serve(const struct gguf_context *ctx, const char *path) {
    struct tokenizer *tk = tokenizer_init(ctx);
    if (!tk) { fprintf(stderr, "tokenizer init failed\n"); return; }
    struct model m;
    if (model_init(&m, ctx) != 0) { tokenizer_free(tk); return; }
    struct kvcache kv;
    if (kvcache_init(&kv, &m, SERVE_SEQ) != 0) { model_free(&m); tokenizer_free(tk); return; }
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
            if (recv_line(c, line, sizeof line) < 0) break;
            // first turn opens the conversation (BOS); later ones first close
            // the previous model turn, whose <turn|> was never fed back
            snprintf(chat, sizeof chat,
                     pos == 0 ? "<|turn>user\n%s<turn|>\n<|turn>model\n"
                              : "<turn|>\n<|turn>user\n%s<turn|>\n<|turn>model\n", line);
            int n = tokenizer_encode(tk, chat, promptv, 4096);
            int skip = pos == 0 ? 0 : 1;                 // tokenizer_encode always prepends BOS
            if (n <= skip || pos + (n - skip) + 1 >= SERVE_SEQ) { fprintf(stderr, "context full\n"); break; }
            clock_t t0 = clock();
            for (int i = skip; i + 1 < n; i++) model_prefill(&m, &kv, promptv[i], pos++);
            int best = model_forward_next(&m, &kv, promptv[n - 1], pos++);
            int g = 0, fail = 0;
            for (;; g++) {                               // stream raw token text, turn end included
                if (client_reset(c) || send_piece(c, tokenizer_token_text(tk, best)) != 0) { fail = 1; break; }
                if (best == eot || best == eos || pos + 1 >= SERVE_SEQ) break;
                best = model_forward_next(&m, &kv, best, pos++);
            }
            double dt = (double)(clock() - t0) / CLOCKS_PER_SEC;
            fprintf(stderr, "turn: %d in, %d out, %.2fs (%.2f tok/s)\n",
                    n - skip, g + 1, dt, (n - skip + g + 1) / (dt > 0 ? dt : 1e-9));
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
    clock_t tp = clock();
    for (int i = 0; i + 1 < n_prompt; i++) model_prefill(&m, &kv, promptv[i], pos++);
    int best = model_forward_next(&m, &kv, promptv[n_prompt - 1], pos++);
    double t_prompt = (double)(clock() - tp) / CLOCKS_PER_SEC;

    // greedy generation
    clock_t tg = clock();
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
    double t_gen = (double)(clock() - tg) / CLOCKS_PER_SEC;

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
    const char *model = NULL, *prompt = NULL, *spath = NULL, *cpath = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc)      model  = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) spath  = argv[++i];
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) cpath  = argv[++i];
    }
    if (cpath) { client(cpath); return 0; }              // client mode needs no model
    if (!model) {
        printf("Usage: %s -m <model.gguf> [-p \"prompt\" | -s <socket>]\n"
               "       %s -c <socket>\n", argv[0], argv[0]);
        return 1;
    }

    struct gguf_context *ctx = load_gguf(model);
    if (!ctx) return 1;

    gguf_dump(ctx);

    struct config cfg;
    if (config_load(&cfg, ctx) == 0) { printf("\n"); config_print(&cfg); }

    if (spath)       serve(ctx, spath);
    else if (prompt) { printf("\n"); generate(ctx, prompt); }

    free_gguf(ctx);
    return 0;
}
