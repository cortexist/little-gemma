#include "decision-head.h"
#include "gguf.h"

#include <stdio.h>
#include <string.h>

/* This adapter is deliberately separate from the numerical core. A host with
 * its own GGUF reader can construct the same spec without linking this file. */
static const struct gguf_meta *meta(const struct gguf_context *g, const char *key, uint32_t type) {
    const struct gguf_meta *m = gguf_find_meta(g, key);
    return m && m->type == type ? m : NULL;
}

static const float *tensor(const struct gguf_context *g, const char *name,
                           size_t rows, size_t cols, int matrix) {
    const struct gguf_tensor *t = gguf_find_tensor(g, name);
    size_t bytes = rows * cols * sizeof(float);
    if (!t || t->type != 0 /* GGML F32 */ || t->n_dims != (uint32_t)(matrix ? 2 : 1) ||
        t->dims[0] != cols || (matrix && t->dims[1] != rows) ||
        t->offset > g->data_size || bytes > g->data_size - (size_t)t->offset ||
        !t->data || (uintptr_t)t->data % _Alignof(float)) return NULL;
    return t->data;
}

struct decision_head *decision_head_load_gguf(const char *path, char *error, size_t error_size) {
    const char *why = "Invalid decision-head GGUF";
    struct decision_head *head = NULL;
    struct gguf_context *g = NULL;
    struct gguf_header header;
    FILE *f = path ? fopen(path, "rb") : NULL;
    if (!f) { why = "Cannot open decision-head GGUF"; goto done; }
    int ok = fread(&header, 1, sizeof(header), f) == sizeof(header);
    fclose(f);
    if (!ok || header.magic != GGUF_MAGIC_LE || header.version != 3 ||
        header.num_meta > 256 || header.num_tensors > 130) {
        why = "Invalid GGUF header or excessive decision-head entry count"; goto done;
    }
    g = load_gguf(path);
    if (!g) goto done;
    for (uint64_t i = 0; i < g->header.num_meta; i++)
        for (uint64_t j = 0; j < i; j++) if (!strcmp(g->meta[i].key, g->meta[j].key)) {
            why = "Duplicate decision-head metadata key"; goto done;
        }
    for (uint64_t i = 0; i < g->header.num_tensors; i++)
        for (uint64_t j = 0; j < i; j++) if (!strcmp(g->tensors[i].name, g->tensors[j].name)) {
            why = "Duplicate decision-head tensor name"; goto done;
        }
    struct decision_head_spec s = {0};
    const struct gguf_meta *m;
#define GET(key, kind) do { m = meta(g, key, kind); if (!m) { why = "Missing/incorrectly typed " key; goto done; } } while (0)
#define TEXT(key, dst) do { GET(key, GGUF_TYPE_STRING); dst = m->value.str; } while (0)
#define COUNT(key, dst) do { GET(key, GGUF_TYPE_UINT32); dst = m->value.u32; } while (0)
#define REQUIRE(key, expected) do { GET(key, GGUF_TYPE_STRING); if (strcmp(m->value.str, expected)) { why = "Unsupported " key; goto done; } } while (0)
    REQUIRE("general.architecture", "decision_head");
    GET("decision_head.version", GGUF_TYPE_UINT32);
    if (m->value.u32 != 1) { why = "Unsupported decision-head schema version"; goto done; }
    GET("decision_head.architecture", GGUF_TYPE_STRING);
    if (!strcmp(m->value.str, "linear")) s.architecture = DECISION_HEAD_LINEAR;
    else if (!strcmp(m->value.str, "mlp")) s.architecture = DECISION_HEAD_MLP;
    else { why = "Unsupported decision-head architecture"; goto done; }
    REQUIRE("decision_head.activation", s.architecture == DECISION_HEAD_MLP ? "tanh" : "identity");
    REQUIRE("decision_head.normalization", "standardize_clip");
    REQUIRE("decision_head.ensemble_reduction", "mean_logits");
    REQUIRE("decision_head.output_transform", "temperature_softmax");
    COUNT("decision_head.input_dim", s.features.input_dim);
    COUNT("decision_head.hidden_dim", s.hidden_dim);
    COUNT("decision_head.output_dim", s.output_dim);
    COUNT("decision_head.ensemble_size", s.ensemble_size);
    size_t in = s.features.input_dim, h = s.hidden_dim, out = s.output_dim, e = s.ensemble_size;
    if (!in || in > 65536 || !out || out > 4096 || !e || e > 32 || h > 4096 ||
        (s.architecture == DECISION_HEAD_LINEAR ? h != 0 : !h) ||
        (uint64_t)e * ((uint64_t)in*h + h + (uint64_t)out*(h ? h : in) + out) > 64ull*1024*1024) {
        why = "Invalid/oversized decision-head dimensions"; goto done;
    }
    if (g->header.num_tensors != 2 + e * (h ? 4 : 2)) {
        why = "Unexpected decision-head tensor count"; goto done;
    }
    GET("decision_head.clip", GGUF_TYPE_FLOAT32); s.clip = m->value.f32;
    GET("decision_head.temperature", GGUF_TYPE_FLOAT64); s.temperature = m->value.f64;
    TEXT("decision_head.task", s.task);
    TEXT("decision_head.feature.model_sha256", s.features.model_sha256);
    TEXT("decision_head.feature.input_format", s.features.input_format);
    TEXT("decision_head.feature.capture_boundary", s.features.capture_boundary);
    GET("decision_head.feature.layer", GGUF_TYPE_INT32); s.features.layer = m->value.i32;
    GET("decision_head.feature.answer_channel_seed", GGUF_TYPE_BOOL);
    s.features.answer_channel_seed = m->value.bool_;
    TEXT("decision_head.source_contract", s.source_contract_json);
    GET("decision_head.labels", GGUF_TYPE_ARRAY);
    if (m->value.arr.type != GGUF_TYPE_STRING || m->value.arr.n != out) {
        why = "Decision-head label count/type mismatch"; goto done;
    }
    s.labels = (const char *const *)m->value.arr.data;
    s.mean = tensor(g, "input.mean", 1, in, 0);
    s.scale = tensor(g, "input.scale", 1, in, 0);
    struct decision_head_member members[32] = {{0}};
    s.members = members;
    if (!s.mean || !s.scale) { why = "Invalid decision-head normalization tensors"; goto done; }
    for (size_t i = 0; i < e; i++) {
        char name[80];
#define WEIGHT(suffix, field, rows, cols, matrix) do { \
    snprintf(name, sizeof name, "member.%zu." suffix, i); \
    members[i].field = tensor(g, name, rows, cols, matrix); \
    if (!members[i].field) { why = "Missing, malformed or truncated decision-head weight tensor"; goto done; } \
} while (0)
        if (h) {
            WEIGHT("hidden.weight", hidden_weight, h, in, 1);
            WEIGHT("hidden.bias", hidden_bias, 1, h, 0);
        }
        WEIGHT("output.weight", output_weight, out, h ? h : in, 1);
        WEIGHT("output.bias", output_bias, 1, out, 0);
#undef WEIGHT
    }
    head = decision_head_create(&s, error, error_size);
    free_gguf(g);
    return head;
done:
    if (error && error_size) snprintf(error, error_size, "%s", why);
    free_gguf(g); return NULL;
#undef REQUIRE
#undef COUNT
#undef TEXT
#undef GET
}
