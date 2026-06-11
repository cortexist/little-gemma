#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gguf.h"
#include "quant.h"  // ggml_type_name / ggml_nbytes for the tensor dump

// ---- platform helpers: 64-bit file size/seek and available memory ---------

#ifdef _WIN32
#include <windows.h>

static int64_t file_size64(FILE *f) {
    if (_fseeki64(f, 0, SEEK_END) != 0) return -1;
    return _ftelli64(f);
}
static int seek64(FILE *f, int64_t off) {
    return _fseeki64(f, off, SEEK_SET);
}
// Bytes of physical RAM currently available, or 0 if it can't be determined.
static size_t available_memory(void) {
    MEMORYSTATUSEX s;
    s.dwLength = sizeof(s);
    return GlobalMemoryStatusEx(&s) ? (size_t)s.ullAvailPhys : 0;
}

#else
#include <unistd.h>
#include <fcntl.h>   // posix_fadvise

static int64_t file_size64(FILE *f) {
    if (fseeko(f, 0, SEEK_END) != 0) return -1;
    return (int64_t)ftello(f);
}
static int seek64(FILE *f, int64_t off) {
    return fseeko(f, (off_t)off, SEEK_SET);
}
// MemAvailable counts reclaimable page cache; _SC_AVPHYS_PAGES counts only
// FREE pages, so right after a 7 GB model file is read it reports near zero
// and the NEXT load — a 175 MB mmproj on a 16 GB Orin — gets refused.
static size_t available_memory(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[128];
        unsigned long long kb;
        while (fgets(line, sizeof line, f)) {
            if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
                fclose(f);
                return (size_t)kb * 1024;
            }
        }
        fclose(f);
    }
#if defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long psize = sysconf(_SC_PAGESIZE);
    if (pages > 0 && psize > 0) return (size_t)pages * (size_t)psize;
#endif
    return 0; // unknown
}
#endif

// Optional hard cap on the tensor-data allocation (0 = none). See gguf.h.
static size_t g_max_data_bytes = 0;

void gguf_set_max_data_bytes(size_t max_bytes) {
    g_max_data_bytes = max_bytes;
}

// Decide whether `need` bytes may be loaded. Prints the reason and returns
// false if it exceeds the configured cap or available physical memory.
static bool memory_ok(size_t need) {
    if (g_max_data_bytes && need > g_max_data_bytes) {
        fprintf(stderr,
            "Tensor data (%zu bytes) exceeds the configured limit (%zu bytes).\n",
            need, g_max_data_bytes);
        return false;
    }
    size_t avail = available_memory();
    if (avail && need > avail) {
        fprintf(stderr,
            "Tensor data (%zu bytes) exceeds available memory (%zu bytes); "
            "refusing to load to avoid paging to disk.\n", need, avail);
        return false;
    }
    return true;
}

static const size_t GGUF_TYPE_SIZES[GGUF_TYPE_COUNT] = {
    [GGUF_TYPE_UINT8]   = sizeof(uint8_t),
    [GGUF_TYPE_INT8]    = sizeof(int8_t),
    [GGUF_TYPE_UINT16]  = sizeof(uint16_t),
    [GGUF_TYPE_INT16]   = sizeof(int16_t),
    [GGUF_TYPE_UINT32]  = sizeof(uint32_t),
    [GGUF_TYPE_INT32]   = sizeof(int32_t),
    [GGUF_TYPE_FLOAT32] = sizeof(float),
    [GGUF_TYPE_BOOL]    = sizeof(int8_t),
    [GGUF_TYPE_STRING]  = 0, // variable length
    [GGUF_TYPE_ARRAY]   = 0, // variable length
    [GGUF_TYPE_UINT64]  = sizeof(uint64_t),
    [GGUF_TYPE_INT64]   = sizeof(int64_t),
    [GGUF_TYPE_FLOAT64] = sizeof(double),
};

static const char *GGUF_TYPE_NAMES[GGUF_TYPE_COUNT] = {
    [GGUF_TYPE_UINT8]   = "u8",
    [GGUF_TYPE_INT8]    = "i8",
    [GGUF_TYPE_UINT16]  = "u16",
    [GGUF_TYPE_INT16]   = "i16",
    [GGUF_TYPE_UINT32]  = "u32",
    [GGUF_TYPE_INT32]   = "i32",
    [GGUF_TYPE_FLOAT32] = "f32",
    [GGUF_TYPE_BOOL]    = "bool",
    [GGUF_TYPE_STRING]  = "str",
    [GGUF_TYPE_ARRAY]   = "arr",
    [GGUF_TYPE_UINT64]  = "u64",
    [GGUF_TYPE_INT64]   = "i64",
    [GGUF_TYPE_FLOAT64] = "f64",
};

size_t gguf_type_size(uint32_t type) {
    return type < GGUF_TYPE_COUNT ? GGUF_TYPE_SIZES[type] : 0;
}

const char *gguf_type_name(uint32_t type) {
    const char *n = type < GGUF_TYPE_COUNT ? GGUF_TYPE_NAMES[type] : NULL;
    return n ? n : "?";
}

// Guard against absurd lengths from corrupt/malicious files (1 GiB).
#define GGUF_MAX_LEN (1ull << 30)

static bool read_exact(FILE *f, void *dst, size_t size) {
    return size == 0 || fread(dst, 1, size, f) == size;
}

// Read a GGUF string: u64 length followed by `length` raw bytes (no NUL on disk).
// Returns a malloc'd, NUL-terminated copy, or NULL on error.
static char *read_string(FILE *f) {
    uint64_t len;
    if (!read_exact(f, &len, sizeof(len))) return NULL;
    if (len > GGUF_MAX_LEN)               return NULL;

    char *str = malloc(len + 1);
    if (!str) return NULL;

    if (!read_exact(f, str, len)) {
        free(str);
        return NULL;
    }
    str[len] = '\0';
    return str;
}

// Read one value of the given scalar type into the kv union.
// Does NOT handle STRING or ARRAY (callers do). Returns false on error.
static bool read_scalar(FILE *f, uint32_t type, struct gguf_kv *kv) {
    size_t size = gguf_type_size(type);
    if (size == 0) return false; // not a fixed-size scalar
    return read_exact(f, &kv->value, size);
}

// Free everything a kv owns. Safe on a zero-initialized kv.
static void free_kv(struct gguf_kv *kv) {
    free(kv->key);
    if (kv->type == GGUF_TYPE_STRING) {
        free(kv->value.str);
    } else if (kv->type == GGUF_TYPE_ARRAY) {
        if (kv->value.arr.type == GGUF_TYPE_STRING && kv->value.arr.data) {
            char **items = kv->value.arr.data;
            for (uint64_t j = 0; j < kv->value.arr.n; j++) free(items[j]);
        }
        free(kv->value.arr.data);
    }
}

static bool read_array(FILE *f, struct gguf_kv *kv) {
    uint32_t et;
    uint64_t n;
    if (!read_exact(f, &et, sizeof(et))) return false;
    if (!read_exact(f, &n,  sizeof(n)))  return false;
    if (et == GGUF_TYPE_ARRAY)           return false; // nested arrays are illegal

    kv->value.arr.type = et;
    kv->value.arr.n    = n;
    kv->value.arr.data = NULL;
    if (n == 0) return true;

    if (et == GGUF_TYPE_STRING) {
        char **items = calloc(n, sizeof(char *));
        if (!items) return false;
        kv->value.arr.data = items; // assign first so free_kv can clean partial reads
        for (uint64_t j = 0; j < n; j++) {
            items[j] = read_string(f);
            if (!items[j]) return false;
        }
        return true;
    }

    size_t esize = gguf_type_size(et);
    if (esize == 0)             return false;          // unknown element type
    if (n > GGUF_MAX_LEN / esize) return false;        // overflow / absurd size
    void *buf = malloc(n * esize);
    if (!buf) return false;
    kv->value.arr.data = buf;
    return read_exact(f, buf, n * esize);
}

static bool read_kv(FILE *f, struct gguf_kv *kv) {
    kv->key = read_string(f);
    if (!kv->key) return false;
    if (!read_exact(f, &kv->type, sizeof(kv->type))) return false;

    switch (kv->type) {
        case GGUF_TYPE_STRING:
            kv->value.str = read_string(f);
            return kv->value.str != NULL;
        case GGUF_TYPE_ARRAY:
            return read_array(f, kv);
        default:
            return read_scalar(f, kv->type, kv);
    }
}

static bool read_tensor(FILE *f, struct gguf_tensor *t) {
    t->name = read_string(f);
    if (!t->name) return false;
    if (!read_exact(f, &t->n_dims, sizeof(t->n_dims))) return false;
    if (t->n_dims < 1 || t->n_dims > 4) return false; // dims[4] is fixed-size
    if (!read_exact(f, t->dims, sizeof(uint64_t) * t->n_dims)) return false;
    if (!read_exact(f, &t->type, sizeof(t->type)))     return false;
    if (!read_exact(f, &t->offset, sizeof(t->offset))) return false;
    return true;
}

// Pull general.alignment out of the metadata if present (default 32).
static size_t find_alignment(const struct gguf_context *ctx) {
    for (uint64_t i = 0; i < ctx->header.num_kv; i++) {
        const struct gguf_kv *kv = &ctx->kv[i];
        if (kv->type == GGUF_TYPE_UINT32 &&
            strcmp(kv->key, "general.alignment") == 0) {
            return kv->value.u32 ? kv->value.u32 : 32;
        }
    }
    return 32;
}

struct gguf_context *load_gguf(const char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        perror("Failed to open file");
        return NULL;
    }

    struct gguf_context *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        fclose(f);
        return NULL;
    }

    if (!read_exact(f, &ctx->header, sizeof(ctx->header))) {
        fprintf(stderr, "Failed to read header.\n");
        goto fail;
    }
    if (ctx->header.magic == GGUF_MAGIC_BE) {
        fprintf(stderr, "Big-endian GGUF files are not supported (this reader assumes little-endian).\n");
        goto fail;
    }
    if (ctx->header.magic != GGUF_MAGIC_LE) {
        fprintf(stderr, "Invalid GGUF magic number.\n");
        goto fail;
    }
    if (ctx->header.version != 2 && ctx->header.version != 3) {
        fprintf(stderr, "Unsupported GGUF version %u (only 2 and 3 are handled).\n",
                ctx->header.version);
        goto fail;
    }

    if (ctx->header.num_kv) {
        ctx->kv = calloc(ctx->header.num_kv, sizeof(*ctx->kv));
        if (!ctx->kv) goto fail;
    }
    for (uint64_t i = 0; i < ctx->header.num_kv; i++) {
        if (!read_kv(f, &ctx->kv[i])) {
            fprintf(stderr, "Failed to read KV pair %llu.\n", (unsigned long long)i);
            goto fail;
        }
    }

    if (ctx->header.num_tensors) {
        ctx->tensors = calloc(ctx->header.num_tensors, sizeof(*ctx->tensors));
        if (!ctx->tensors) goto fail;
    }
    for (uint64_t i = 0; i < ctx->header.num_tensors; i++) {
        if (!read_tensor(f, &ctx->tensors[i])) {
            fprintf(stderr, "Failed to read tensor info %llu.\n", (unsigned long long)i);
            goto fail;
        }
    }

    ctx->alignment = find_alignment(ctx);

    // Tensor data starts at the next `alignment` boundary after the info section.
    long pos = ftell(f);
    if (pos < 0) {
        fprintf(stderr, "Failed to locate tensor data section.\n");
        goto fail;
    }
    size_t a = ctx->alignment;
    ctx->data_offset = ((size_t)pos + a - 1) / a * a;

    // Size of the data section = file size - data_offset.
    int64_t fsize = file_size64(f);
    if (fsize < 0 || (uint64_t)fsize < ctx->data_offset) {
        fprintf(stderr, "Invalid file size relative to data offset.\n");
        goto fail;
    }
    ctx->data_size = (size_t)((uint64_t)fsize - ctx->data_offset);

    // Refuse to load if it won't fit (cap / physical memory) rather than page.
    if (!memory_ok(ctx->data_size)) goto fail;

    // Eagerly read the whole data section into memory.
    if (ctx->data_size) {
        ctx->data = malloc(ctx->data_size);
        if (!ctx->data) {
            fprintf(stderr, "Out of memory: could not allocate %zu bytes for tensor data.\n",
                    ctx->data_size);
            goto fail;
        }
        if (seek64(f, (int64_t)ctx->data_offset) != 0 ||
            fread(ctx->data, 1, ctx->data_size, f) != ctx->data_size) {
            fprintf(stderr, "Failed to read tensor data.\n");
            goto fail;
        }
#ifndef _WIN32
        // The read just left a second copy of the model in the page cache —
        // on a unified-memory board that is real pressure (a 7 GB model holds
        // 14 GB of a 16 GB Orin until the kernel reclaims). We own the only
        // copy that matters now; tell the kernel to drop the cached one.
        posix_fadvise(fileno(f), 0, 0, POSIX_FADV_DONTNEED);
#endif
    }

    // Point each tensor into the loaded buffer (offsets are relative to data_offset).
    for (uint64_t i = 0; i < ctx->header.num_tensors; i++) {
        struct gguf_tensor *t = &ctx->tensors[i];
        // Bounds-check the start; full length needs ggml type traits (TODO).
        if (t->offset > ctx->data_size) {
            fprintf(stderr, "Tensor '%s' offset out of bounds.\n", t->name);
            goto fail;
        }
        t->data = (unsigned char *)ctx->data + t->offset;
    }

    fclose(f);
    return ctx;

fail:
    if (f) fclose(f);
    free_gguf(ctx);
    return NULL;
}

void free_gguf(struct gguf_context *ctx) {
    if (!ctx) return;
    free(ctx->data);
    if (ctx->kv) {
        for (uint64_t i = 0; i < ctx->header.num_kv; i++) free_kv(&ctx->kv[i]);
        free(ctx->kv);
    }
    if (ctx->tensors) {
        for (uint64_t i = 0; i < ctx->header.num_tensors; i++) free(ctx->tensors[i].name);
        free(ctx->tensors);
    }
    free(ctx);
}

// ---- pretty-printing ------------------------------------------------------

static void print_scalar(const struct gguf_kv *kv, uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:   printf("%u",   kv->value.u8);  break;
        case GGUF_TYPE_INT8:    printf("%d",   kv->value.i8);  break;
        case GGUF_TYPE_UINT16:  printf("%u",   kv->value.u16); break;
        case GGUF_TYPE_INT16:   printf("%d",   kv->value.i16); break;
        case GGUF_TYPE_UINT32:  printf("%u",   kv->value.u32); break;
        case GGUF_TYPE_INT32:   printf("%d",   kv->value.i32); break;
        case GGUF_TYPE_FLOAT32: printf("%g",   kv->value.f32); break;
        case GGUF_TYPE_FLOAT64: printf("%g",   kv->value.f64); break;
        case GGUF_TYPE_BOOL:    printf("%s",   kv->value.bool_ ? "true" : "false"); break;
        case GGUF_TYPE_UINT64:  printf("%llu", (unsigned long long)kv->value.u64); break;
        case GGUF_TYPE_INT64:   printf("%lld", (long long)kv->value.i64);          break;
        default:                printf("?"); break;
    }
}

// Read element `idx` out of a scalar array buffer into a temporary kv and print it.
static void print_array_element(const struct gguf_kv *arr, uint64_t idx) {
    uint32_t et = arr->value.arr.type;
    size_t esize = gguf_type_size(et);
    struct gguf_kv tmp = {0};
    memcpy(&tmp.value, (const char *)arr->value.arr.data + idx * esize, esize);
    print_scalar(&tmp, et);
}

// Print a string, truncating to at most `max` UTF-8 characters (codepoints) so
// we never cut a multibyte sequence in half. Appends "..." when truncated.
static void print_string(const char *s, size_t max) {
    if (!s) { printf("(null)"); return; }
    const unsigned char *p = (const unsigned char *)s;
    size_t width = 0;                     // rendered (visible) characters emitted
    while (*p) {
        if ((*p & 0xC0) == 0x80) {        // UTF-8 continuation byte: part of the
            putchar(*p);                  // current codepoint, emit raw, no budget cost
            p++;
            continue;
        }
        // Lead byte: a new character. Escaped control chars render as two glyphs.
        size_t w = (*p == '\n' || *p == '\r' || *p == '\t') ? 2 : 1;
        if (width + w > max) break;
        width += w;
        switch (*p) {                     // keep the value on one line
            case '\n': fputs("\\n", stdout); break;
            case '\r': fputs("\\r", stdout); break;
            case '\t': fputs("\\t", stdout); break;
            default:   putchar(*p);          break;
        }
        p++;
    }
    if (*p) printf("...");
}

#define GGUF_STR_PREVIEW 40

static void print_value(const struct gguf_kv *kv) {
    if (kv->type == GGUF_TYPE_STRING) {
        print_string(kv->value.str, GGUF_STR_PREVIEW);
        return;
    }
    if (kv->type == GGUF_TYPE_ARRAY) {
        uint64_t n = kv->value.arr.n;
        uint32_t et = kv->value.arr.type;
        printf("[%s; %llu] ", gguf_type_name(et), (unsigned long long)n);
        uint64_t show = n < 4 ? n : 4; // preview the first few
        printf("{");
        for (uint64_t j = 0; j < show; j++) {
            if (j) printf(", ");
            if (et == GGUF_TYPE_STRING) {
                char **items = kv->value.arr.data;
                print_string(items[j], GGUF_STR_PREVIEW);
                break;
            } 
            print_array_element(kv, j);
        }
        if (n > show) printf(", ...");
        printf("}");
        return;
    }
    print_scalar(kv, kv->type);
}

void gguf_dump(const struct gguf_context *ctx) {
    if (!ctx) return;

    printf("--- header ---\n");
    printf("version: %u\n", ctx->header.version);
    printf("tensors: %llu\n", (unsigned long long)ctx->header.num_tensors);
    printf("kv pairs: %llu\n", (unsigned long long)ctx->header.num_kv);
    printf("alignment: %zu\n", ctx->alignment);
    printf("data offset: %zu\n", ctx->data_offset);
    printf("data: %zu bytes (in memory)\n", ctx->data_size);

    printf("\n--- metadata ---\n");
    for (uint64_t i = 0; i < ctx->header.num_kv; i++) {
        const struct gguf_kv *kv = &ctx->kv[i];
        printf("%s (%s) = ", kv->key, gguf_type_name(kv->type));
        print_value(kv);
        printf("\n");
    }

    printf("\n--- tensors ---\n");
    for (uint64_t i = 0; i < ctx->header.num_tensors; i++) {
        const struct gguf_tensor *t = &ctx->tensors[i];
        printf("%s | dims=[", t->name);
        for (uint32_t d = 0; d < t->n_dims; d++) {
            if (d) printf(", ");
            printf("%llu", (unsigned long long)t->dims[d]);
        }
        int64_t n = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) n *= (int64_t)t->dims[d];
        printf("] | type=%s | offset=%llu | nbytes=%zu\n",
               ggml_type_name(t->type), (unsigned long long)t->offset,
               ggml_nbytes(t->type, n));
    }
}

// ---- lookup ---------------------------------------------------------------

const struct gguf_kv *gguf_find_kv(const struct gguf_context *ctx, const char *key) {
    for (uint64_t i = 0; i < ctx->header.num_kv; i++) {
        if (strcmp(ctx->kv[i].key, key) == 0) return &ctx->kv[i];
    }
    return NULL;
}

const struct gguf_tensor *gguf_find_tensor(const struct gguf_context *ctx, const char *name) {
    for (uint64_t i = 0; i < ctx->header.num_tensors; i++) {
        if (strcmp(ctx->tensors[i].name, name) == 0) return &ctx->tensors[i];
    }
    return NULL;
}

uint32_t gguf_get_u32(const struct gguf_context *ctx, const char *key, uint32_t fallback) {
    const struct gguf_kv *kv = gguf_find_kv(ctx, key);
    if (!kv) return fallback;
    switch (kv->type) {
        case GGUF_TYPE_UINT32: return kv->value.u32;
        case GGUF_TYPE_INT32:  return (uint32_t)kv->value.i32;
        default:               return fallback;
    }
}

int32_t gguf_get_i32(const struct gguf_context *ctx, const char *key, int32_t fallback) {
    const struct gguf_kv *kv = gguf_find_kv(ctx, key);
    if (!kv) return fallback;
    switch (kv->type) {
        case GGUF_TYPE_INT32:  return kv->value.i32;
        case GGUF_TYPE_UINT32: return (int32_t)kv->value.u32;
        default:               return fallback;
    }
}

float gguf_get_f32(const struct gguf_context *ctx, const char *key, float fallback) {
    const struct gguf_kv *kv = gguf_find_kv(ctx, key);
    if (!kv || kv->type != GGUF_TYPE_FLOAT32) return fallback;
    return kv->value.f32;
}

const char *gguf_get_str(const struct gguf_context *ctx, const char *key, const char *fallback) {
    const struct gguf_kv *kv = gguf_find_kv(ctx, key);
    if (!kv || kv->type != GGUF_TYPE_STRING) return fallback;
    return kv->value.str;
}
