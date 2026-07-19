#ifndef GGUF_H
#define GGUF_H

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

// "GGUF" (0x47 0x47 0x55 0x46) read as a uint32. This reader assumes a
// little-endian host and little-endian files (the GGUF norm), where the magic
// reads back as 0x46554747. The 0x47475546 value means the byte order is
// opposite our host; we detect it only to reject such files with a clear error.
#define GGUF_MAGIC_LE 0x46554747u
#define GGUF_MAGIC_BE 0x47475546u

// GGUF metadata value types (also used as array element types).
enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
    GGUF_TYPE_COUNT,
};

struct gguf_header {
    uint32_t magic;
    uint32_t version;
    uint64_t num_tensors;
    uint64_t num_meta;
};

// A single metadata key/value pair.
// `type` is the tag; it selects which member of `value` is live.
// Scalars are stored inline (no allocation); only STRING and ARRAY own heap memory.
struct gguf_meta {
    char    *key;        // owned
    uint32_t type;       // enum gguf_type
    union {
        uint8_t  u8;
        int8_t   i8;
        uint16_t u16;
        int16_t  i16;
        uint32_t u32;
        int32_t  i32;
        uint64_t u64;
        int64_t  i64;
        float    f32;
        double   f64;
        bool     bool_;
        char    *str;            // GGUF_TYPE_STRING — owned
        struct {                 // GGUF_TYPE_ARRAY
            uint32_t type;       // element type (enum gguf_type, never ARRAY)
            uint64_t n;          // element count
            void    *data;       // n elements of `type`;
                                 // if type == STRING, this is char*[n] (each owned)
        } arr;
    } value;
};

struct gguf_tensor {
    char    *name;       // owned
    uint32_t n_dims;     // 1..4
    uint64_t dims[4];
    uint32_t type;       // tensor type
    uint64_t offset;     // byte offset into the tensor-data blob
    void    *data;       // pointer into ctx->data (ctx->data + offset),
                         // or NULL if there is no data section
};

struct gguf_context {
    struct gguf_header  header;
    struct gguf_meta   *meta;        // header.num_meta entries
    struct gguf_tensor *tensors;   // header.num_tensors entries
    size_t              alignment; // general.alignment (default 32)
    size_t              data_offset;

    // The tensor-data section: on POSIX a read-only file mapping (pages are
    // clean and evictable — weights a backend has copied elsewhere cost no
    // RAM once cold), on Windows an eagerly-read heap buffer. Tensors point
    // into it either way.
    void   *data;                  // the data section (into map_base, or owned heap)
    size_t  data_size;             // its size in bytes (file_size - data_offset)
    void   *map_base;              // mmap base when file-mapped (NULL = heap)
    size_t  map_len;               // mapped length incl. the alignment head
    int     map_fd;                // kept open for gguf_data_dontneed (-1 = none)
};

// Returns the on-disk size in bytes of a scalar value type, or 0 for
// variable-length types (STRING, ARRAY) and out-of-range types.
size_t gguf_type_size(uint32_t type);
const char *gguf_type_name(uint32_t type);

// Parse a GGUF file (header, metadata, tensor info) and load the tensor data
// section into memory. Returns NULL on any error, including when the data would
// not fit in available memory (or exceeds the configured limit), so loading
// never silently pages to disk. Free with free_gguf().
struct gguf_context *load_gguf(const char *filepath);
void free_gguf(struct gguf_context *ctx);

// Hand the whole pages inside [p, p+bytes) of the mmap'd data section back
// to the OS — the process mapping and the page cache both. For weights a
// backend has copied elsewhere the blob bytes are dead, and on a unified-
// memory board returning them as the copies are made is what lets model +
// copies coexist. A later read refaults from the file: this is a memory
// hint, never a correctness event. No-op for heap-loaded data (Windows) and
// for ranges outside this context's data section.
void gguf_data_dontneed(const struct gguf_context *ctx, const void *p, size_t bytes);

// Set an upper bound (bytes) on the tensor-data allocation; load_gguf() errors
// out if the data section is larger. 0 (the default) means "no explicit cap" —
// only available physical memory is enforced. Affects subsequent load_gguf calls.
void gguf_set_max_data_bytes(size_t max_bytes);

// Print a human-readable summary of a parsed context.
void gguf_dump(const struct gguf_context *ctx);

// ---- lookup ---------------------------------------------------------------

// Find a metadata pair / tensor by name. NULL if absent.
const struct gguf_meta   *gguf_find_meta(const struct gguf_context *ctx, const char *key);
const struct gguf_tensor *gguf_find_tensor(const struct gguf_context *ctx, const char *name);

// Typed metadata accessors. Return `fallback` if the key is missing or the value
// is not (convertible to) the requested type. Integer getters accept UINT32/INT32.
uint32_t    gguf_get_u32(const struct gguf_context *ctx, const char *key, uint32_t fallback);
int32_t     gguf_get_i32(const struct gguf_context *ctx, const char *key, int32_t fallback);
float       gguf_get_f32(const struct gguf_context *ctx, const char *key, float fallback);
const char *gguf_get_str(const struct gguf_context *ctx, const char *key, const char *fallback);

#endif // GGUF_H
