#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "gguf.h"

int main(int argc, char **argv) {
#ifdef _WIN32
    // GGUF strings (e.g. tokenizer tokens) are UTF-8; tell the console so they
    // render correctly instead of as code-page mojibake.
    SetConsoleOutputCP(CP_UTF8);
#endif

    if (argc < 2) {
        printf("Usage: %s <path/to/model.gguf>\n", argv[0]);
        return 1;
    }

    // Optional cap on the tensor-data allocation (bytes), e.g. GGUF_MAX_DATA_BYTES=1000000000.
    const char *cap = getenv("GGUF_MAX_DATA_BYTES");
    if (cap) gguf_set_max_data_bytes(strtoull(cap, NULL, 10));

    struct gguf_context *ctx = load_gguf(argv[1]);
    if (!ctx) return 1;

    gguf_dump(ctx);
    free_gguf(ctx);
    return 0;
}
