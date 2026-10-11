#ifndef LG_DECISION_HEAD_H
#define LG_DECISION_HEAD_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(DECISION_HEAD_SHARED)
#define DECISION_HEAD_API __declspec(dllexport)
#else
#define DECISION_HEAD_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Independent CPU classifier. No model, tokenizer, CUDA, or GGML dependency
 * in the numerical core. Classes are ordered labels, never hard-coded actions. */
enum decision_head_architecture { DECISION_HEAD_LINEAR, DECISION_HEAD_MLP };

struct decision_head_member {
    /* Row-major [hidden_dim, input_dim], then [output_dim, hidden_dim].
     * Linear heads omit hidden_*; output_weight is [output_dim, input_dim]. */
    const float *hidden_weight, *hidden_bias, *output_weight, *output_bias;
};

struct decision_head_features {
    size_t input_dim;
    const char *model_sha256, *input_format, *capture_boundary;
    int32_t layer;
    int answer_channel_seed;
};

struct decision_head_spec {
    enum decision_head_architecture architecture;
    struct decision_head_features features;
    size_t hidden_dim, output_dim, ensemble_size;
    float clip;
    double temperature;
    const float *mean, *scale;
    const struct decision_head_member *members;
    const char *const *labels;
    const char *task;
    /* Preserved training provenance, not a statement of deployment status. */
    const char *source_contract_json;
};

struct decision_head;
struct decision_head_workspace;

/* Version covers the public structs and function signatures used by FFI hosts. */
DECISION_HEAD_API unsigned decision_head_abi_version(void);

/* Copies all input data; the result is immutable and shareable by threads.
 * Limits: input <= 65536, hidden/output <= 4096, ensemble <= 32,
 * total member parameters <= 64M. All weights must be finite FP32. */
DECISION_HEAD_API struct decision_head *decision_head_create(const struct decision_head_spec *spec,
                                          char *error, size_t error_size);
/* Optional little-gemma GGUF adapter. The core can instead be constructed by
 * another host's loader, e.g. llama.cpp, using decision_head_create(). */
DECISION_HEAD_API struct decision_head *decision_head_load_gguf(const char *path,
                                             char *error, size_t error_size);
DECISION_HEAD_API void decision_head_free(struct decision_head *head);
DECISION_HEAD_API const struct decision_head_spec *decision_head_get_spec(const struct decision_head *head);

/* Hosts must check the feature contract before applying predictions to live
 * behavior. predict() can check vector length, not its causal provenance. */
DECISION_HEAD_API int decision_head_check_features(const struct decision_head *head,
                                const struct decision_head_features *features,
                                char *error, size_t error_size);

/* One workspace per concurrent caller; it belongs to this specific head.
 * The head must outlive its workspaces. predict() allocates no memory. */
DECISION_HEAD_API struct decision_head_workspace *decision_head_workspace_create(const struct decision_head *head);
DECISION_HEAD_API void decision_head_workspace_free(struct decision_head_workspace *workspace);

/* Returns 0 on success, -1 on error. Output buffers each need output_dim
 * floats. On error, choice is SIZE_MAX and output buffers are untouched.
 * Ties choose the first label. Scores are softmax outputs, not a calibration
 * guarantee, authority priority, or permission to execute an action. */
DECISION_HEAD_API int decision_head_predict(const struct decision_head *head,
                          struct decision_head_workspace *workspace,
                          const float *input, size_t input_count,
                          float *logits, float *probabilities, size_t output_capacity,
                          size_t *choice, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif
#endif
