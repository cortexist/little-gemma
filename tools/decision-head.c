#define _POSIX_C_SOURCE 200809L
#include "decision-head.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#endif

static double now(void) {
#ifdef _WIN32
    LARGE_INTEGER t, f; QueryPerformanceCounter(&t); QueryPerformanceFrequency(&f);
    return (double)t.QuadPart/(double)f.QuadPart;
#else
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec*1e-9;
#endif
}

static void json_string(const char *s) {
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p == '"' || *p == '\\') { putchar('\\'); putchar(*p); }
        else if (*p < 32) printf("\\u%04x", (unsigned)*p);
        else putchar(*p);
    }
    putchar('"');
}

int main(int argc, char **argv) {
    if (argc < 3 || (strcmp(argv[1], "inspect") && strcmp(argv[1], "predict")) ||
        (!strcmp(argv[1], "inspect") ? argc != 3 : (argc != 4 && argc != 6))) {
        fprintf(stderr, "Usage: decision-head inspect HEAD.gguf\n"
                        "       decision-head predict HEAD.gguf VECTORS.f32 [--repeat N]\n"
                        "VECTORS.f32: packed little-endian FP32 rows, or - for stdin.\n"
                        "Offline scoring checks dimensions, not vector provenance.\n");
        return 2;
    }
    size_t repeat = 1;
    if (argc == 6) {
        char *end; errno = 0;
        unsigned long n = strtoul(argv[5], &end, 10);
        if (strcmp(argv[4], "--repeat") || !*argv[5] || *end || errno || n < 1 || n > 100000) {
            fprintf(stderr, "Invalid repeat count\n"); return 2;
        }
        repeat = (size_t)n;
    }
    char error[256];
    struct decision_head *head = decision_head_load_gguf(argv[2], error, sizeof error);
    if (!head) { fprintf(stderr, "%s\n", error); return 1; }
    const struct decision_head_spec *s = decision_head_get_spec(head);
    if (!strcmp(argv[1], "inspect")) {
        printf("{\"schema_version\":1,\"architecture\":\"%s\",\"input_dim\":%zu,\"hidden_dim\":%zu,"
               "\"output_dim\":%zu,\"ensemble_size\":%zu,\"temperature\":%.17g,\"task\":",
               s->architecture == DECISION_HEAD_MLP ? "mlp" : "linear", s->features.input_dim,
               s->hidden_dim, s->output_dim, s->ensemble_size, s->temperature);
        json_string(s->task); printf(",\"labels\":[");
        for (size_t i = 0; i < s->output_dim; i++) { if (i) putchar(','); json_string(s->labels[i]); }
        printf("],\"model_sha256\":"); json_string(s->features.model_sha256);
        printf(",\"input_format\":"); json_string(s->features.input_format);
        printf(",\"capture_boundary\":"); json_string(s->features.capture_boundary);
        printf(",\"layer\":%d,\"answer_channel_seed\":%s,\"source_contract_json\":",
               s->features.layer, s->features.answer_channel_seed ? "true" : "false");
        json_string(s->source_contract_json); puts("}");
        decision_head_free(head); return 0;
    }
    /* The container and raw-vector format are little-endian. */
    const uint32_t endian = 1;
    if (*(const unsigned char *)&endian != 1) {
        fprintf(stderr, "Little-endian host required\n"); decision_head_free(head); return 1;
    }
    FILE *input = !strcmp(argv[3], "-") ? stdin : fopen(argv[3], "rb");
#ifdef _WIN32
    if (input == stdin) _setmode(_fileno(stdin), _O_BINARY);
#endif
    struct decision_head_workspace *w = decision_head_workspace_create(head);
    size_t width = s->features.input_dim, classes = s->output_dim;
    float *x = malloc(width*sizeof(float)), *logits = malloc(classes*sizeof(float));
    float *probs = malloc(classes*sizeof(float));
    int status = 1;
    if (!input || !w || !x || !logits || !probs) { fprintf(stderr, "Cannot open input or allocate buffers\n"); goto done; }
    size_t row = 0, n;
    while ((n = fread(x, 1, width*sizeof(float), input)) != 0) {
        if (n != width*sizeof(float)) { fprintf(stderr, "Truncated input vector at row %zu\n", row); goto done; }
        size_t choice;
        /* Excluded warmup when explicitly benchmarking repetitions. */
        if (repeat > 1 && decision_head_predict(head, w, x, width, logits, probs, classes, &choice, error, sizeof error)) {
            fprintf(stderr, "%s\n", error); goto done;
        }
        double start = now();
        for (size_t i = 0; i < repeat; i++)
            if (decision_head_predict(head, w, x, width, logits, probs, classes, &choice, error, sizeof error)) {
                fprintf(stderr, "%s\n", error); goto done;
            }
        double us = (now()-start)*1e6/(double)repeat;
        printf("{\"row\":%zu,\"choice\":%zu,\"label\":", row++, choice); json_string(s->labels[choice]);
        printf(",\"logits\":[");
        for (size_t i = 0; i < classes; i++) printf("%s%.9g", i ? "," : "", (double)logits[i]);
        printf("],\"probabilities\":[");
        for (size_t i = 0; i < classes; i++) printf("%s%.9g", i ? "," : "", (double)probs[i]);
        printf("],\"mean_inference_us\":%.6f,\"repetitions\":%zu}\n", us, repeat);
    }
    if (ferror(input) || !row) { fprintf(stderr, "Empty or unreadable input\n"); goto done; }
    status = 0;
done:
    if (input && input != stdin) fclose(input);
    free(x); free(logits); free(probs); decision_head_workspace_free(w); decision_head_free(head);
    return status;
}
