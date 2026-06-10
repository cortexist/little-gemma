// socket_cat: socat for people who don't have socat.
//
// Connects to an AF_UNIX stream socket and pumps bytes both ways: stdin to
// the socket, socket to stdout, until both sides are done. No protocol, no
// framing — a wire you can type into. Exists because on native Windows the
// usual suspects (socat, nc, even ncat) either don't exist or can't speak
// AF_UNIX, while the OS itself has supported it since Windows 10 1803.
//
//     socket_cat \\path\\to\\model.sock
//
// stdin EOF half-closes the connection (the server sees recv 0 after the
// current turn) but keeps reading, so `echo hi | socket_cat ...` still gets
// the whole streamed answer. The process ends when the server closes.
//
// Two threads, not select(): on Windows a console handle is not a socket and
// no readiness API spans both. One thread blocks on stdin, the other on
// recv — each direction is a dumb copy loop, which is the entire point.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <winsock2.h>   // must precede windows.h
#include <windows.h>
#include <afunix.h>     // AF_UNIX on Windows 10 1803+
typedef SOCKET sock_t;
#define sock_close closesocket
#define SHUT_SEND SD_SEND
static int sock_init(void) { WSADATA w; return WSAStartup(MAKEWORD(2, 2), &w); }
static void spawn(DWORD (WINAPI *fn)(void *), void *arg) { CreateThread(NULL, 0, fn, arg, 0, NULL); }
#define THREAD_FN(name, arg) static DWORD WINAPI name(void *arg)
#define THREAD_RET 0
#else
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
typedef int sock_t;
#define INVALID_SOCKET (-1)
#define sock_close close
#define SHUT_SEND SHUT_WR
// A dead server must surface as a send() error, not a fatal signal.
static int sock_init(void) { signal(SIGPIPE, SIG_IGN); return 0; }
static void spawn(void *(*fn)(void *), void *arg) { pthread_t t; pthread_create(&t, NULL, fn, arg); pthread_detach(t); }
#define THREAD_FN(name, arg) static void *name(void *arg)
#define THREAD_RET NULL
#endif

static int send_all(sock_t s, const char *buf, int n) {
    while (n > 0) {
        int k = (int)send(s, buf, n, 0);
        if (k <= 0) return -1;
        buf += k; n -= k;
    }
    return 0;
}

// stdin -> socket. fgets, not fread: a turn should reach the server when the
// user hits enter, and a console read in text mode already hands us clean \n
// line endings. EOF half-closes so the server finishes the turn in flight.
THREAD_FN(pump_in, arg) {
    sock_t s = *(sock_t *)arg;
    char line[8192];
    while (fgets(line, sizeof line, stdin))
        if (send_all(s, line, (int)strlen(line)) != 0) break;
    shutdown(s, SHUT_SEND);
    return THREAD_RET;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <socket-path>\n", argv[0]);
        return 1;
    }
    struct sockaddr_un sa;
    memset(&sa, 0, sizeof sa);
    sa.sun_family = AF_UNIX;
    sock_t s;
    if (sock_init() != 0 || strlen(argv[1]) >= sizeof sa.sun_path ||
        (s = socket(AF_UNIX, SOCK_STREAM, 0)) == INVALID_SOCKET) {
        fprintf(stderr, "socket setup failed (path too long?)\n"); return 1;
    }
    strcpy(sa.sun_path, argv[1]);
    if (connect(s, (struct sockaddr *)&sa, sizeof sa) != 0) {
        fprintf(stderr, "connect to %s failed\n", argv[1]); return 1;
    }
    spawn(pump_in, &s);

    // socket -> stdout on the main thread, so the process lives exactly as
    // long as the server keeps the connection open. exit() ends the stdin
    // thread too, even if it is parked in a console read.
    char buf[4096];
    int k;
    while ((k = (int)recv(s, buf, sizeof buf, 0)) > 0) {
        fwrite(buf, 1, (size_t)k, stdout);
        fflush(stdout);
    }
    sock_close(s);
    return 0;
}
