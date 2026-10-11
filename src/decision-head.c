#include "decision-head.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct decision_head { struct decision_head_spec spec; };
unsigned decision_head_abi_version(void) { return 1; }
struct decision_head_workspace {
    const struct decision_head *owner;
    float *input, *hidden, *logits, *probabilities;
};

static int fail(char *error, size_t size, const char *message) {
    if (error && size) snprintf(error, size, "%s", message);
    return -1;
}

static char *copy_string(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

static int valid_string(const char *s) { return s && *s && strlen(s) <= 4096; }
static int valid_hash(const char *s) {
    if (!s || strlen(s) != 64) return 0;
    for (size_t i = 0; i < 64; i++)
        if (!((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'a' && s[i] <= 'f'))) return 0;
    return 1;
}

static float *copy_floats(const float *src, size_t count, int positive) {
    if (!src) return NULL;
    for (size_t i = 0; i < count; i++)
        if (!isfinite(src[i]) || (positive && !(src[i] > 0))) return NULL;
    float *dst = malloc(count * sizeof(float));
    if (dst) memcpy(dst, src, count * sizeof(float));
    return dst;
}

void decision_head_free(struct decision_head *head) {
    if (!head) return;
    const struct decision_head_spec *s = &head->spec;
    if (s->members) for (size_t i = 0; i < s->ensemble_size; i++) {
        free((void *)s->members[i].hidden_weight); free((void *)s->members[i].hidden_bias);
        free((void *)s->members[i].output_weight); free((void *)s->members[i].output_bias);
    }
    if (s->labels) for (size_t i = 0; i < s->output_dim; i++) free((void *)s->labels[i]);
    free((void *)s->members); free((void *)s->labels);
    free((void *)s->mean); free((void *)s->scale);
    free((void *)s->features.model_sha256); free((void *)s->features.input_format);
    free((void *)s->features.capture_boundary); free((void *)s->task);
    free((void *)s->source_contract_json); free(head);
}

struct decision_head *decision_head_create(const struct decision_head_spec *src,
                                          char *error, size_t error_size) {
    if (error && error_size) error[0] = 0;
    if (!src) { fail(error, error_size, "Missing decision-head spec"); return NULL; }
    size_t in = src->features.input_dim, h = src->hidden_dim;
    size_t out = src->output_dim, e = src->ensemble_size;
    if (!in || in > 65536 || !out || out > 4096 || !e || e > 32 ||
        (src->architecture != DECISION_HEAD_LINEAR && src->architecture != DECISION_HEAD_MLP) ||
        (src->architecture == DECISION_HEAD_LINEAR ? h != 0 : (!h || h > 4096)) ||
        (uint64_t)e * ((uint64_t)in * h + h + (uint64_t)out * (h ? h : in) + out) > 64ull*1024*1024) {
        fail(error, error_size, "Invalid or oversized decision-head dimensions"); return NULL;
    }
    if (!isfinite(src->clip) || src->clip <= 0 || !isfinite(src->temperature) ||
        src->temperature < FLT_MIN || src->temperature > FLT_MAX || !src->members || !src->labels ||
        !valid_string(src->task) || !valid_hash(src->features.model_sha256) ||
        !valid_string(src->features.input_format) || !valid_string(src->features.capture_boundary) ||
        (src->features.answer_channel_seed != 0 && src->features.answer_channel_seed != 1)) {
        fail(error, error_size, "Invalid decision-head scalars or feature contract"); return NULL;
    }
    for (size_t i = 0; i < out; i++) {
        if (!valid_string(src->labels[i])) {
            fail(error, error_size, "Empty or invalid decision label"); return NULL;
        }
        for (size_t j = 0; j < i; j++) if (!strcmp(src->labels[i], src->labels[j])) {
            fail(error, error_size, "Duplicate decision label"); return NULL;
        }
    }
    struct decision_head *head = calloc(1, sizeof(*head));
    if (!head) { fail(error, error_size, "Out of memory"); return NULL; }
    struct decision_head_spec *s = &head->spec;
    s->architecture = src->architecture; s->hidden_dim = h;
    s->output_dim = out; s->ensemble_size = e; s->clip = src->clip; s->temperature = src->temperature;
    s->features.input_dim = in; s->features.layer = src->features.layer;
    s->features.answer_channel_seed = src->features.answer_channel_seed;
    struct decision_head_member *members = calloc(e, sizeof(*members));
    char **labels = calloc(out, sizeof(*labels));
    s->members = members; s->labels = (const char *const *)labels;
    if (!members || !labels) goto invalid;
    s->features.model_sha256 = copy_string(src->features.model_sha256);
    s->features.input_format = copy_string(src->features.input_format);
    s->features.capture_boundary = copy_string(src->features.capture_boundary);
    s->task = copy_string(src->task);
    s->source_contract_json = copy_string(src->source_contract_json ? src->source_contract_json : "{}");
    if (!s->features.model_sha256 || !s->features.input_format || !s->features.capture_boundary ||
        !s->task || !s->source_contract_json) goto invalid;
    for (size_t i = 0; i < out; i++) if (!(labels[i] = copy_string(src->labels[i]))) goto invalid;
    s->mean = copy_floats(src->mean, in, 0); s->scale = copy_floats(src->scale, in, 1);
    if (!s->mean || !s->scale) goto invalid;
    for (size_t i = 0; i < e; i++) {
        if (h) {
            members[i].hidden_weight = copy_floats(src->members[i].hidden_weight, h * in, 0);
            members[i].hidden_bias = copy_floats(src->members[i].hidden_bias, h, 0);
            if (!members[i].hidden_weight || !members[i].hidden_bias) goto invalid;
        }
        members[i].output_weight = copy_floats(src->members[i].output_weight, out * (h ? h : in), 0);
        members[i].output_bias = copy_floats(src->members[i].output_bias, out, 0);
        if (!members[i].output_weight || !members[i].output_bias) goto invalid;
    }
    return head;
invalid:
    fail(error, error_size, "Invalid/nonfinite tensors, nonpositive scale, or allocation failure");
    decision_head_free(head); return NULL;
}

const struct decision_head_spec *decision_head_get_spec(const struct decision_head *head) {
    return head ? &head->spec : NULL;
}

int decision_head_check_features(const struct decision_head *head,
                                const struct decision_head_features *f,
                                char *error, size_t error_size) {
    if (error && error_size) error[0] = 0;
    if (!head || !f) return fail(error, error_size, "Missing feature contract");
    const struct decision_head_features *want = &head->spec.features;
    if (f->input_dim != want->input_dim || f->layer != want->layer ||
        f->answer_channel_seed != want->answer_channel_seed || !f->model_sha256 ||
        !f->input_format || !f->capture_boundary || strcmp(f->model_sha256, want->model_sha256) ||
        strcmp(f->input_format, want->input_format) || strcmp(f->capture_boundary, want->capture_boundary))
        return fail(error, error_size, "Decision-head feature contract mismatch");
    return 0;
}

void decision_head_workspace_free(struct decision_head_workspace *w) {
    if (!w) return;
    free(w->input); free(w->hidden); free(w->logits); free(w->probabilities); free(w);
}

struct decision_head_workspace *decision_head_workspace_create(const struct decision_head *head) {
    if (!head) return NULL;
    struct decision_head_workspace *w = calloc(1, sizeof(*w));
    if (!w) return NULL;
    w->owner = head;
    w->input = malloc(head->spec.features.input_dim * sizeof(float));
    w->hidden = malloc((head->spec.hidden_dim ? head->spec.hidden_dim : 1) * sizeof(float));
    w->logits = malloc(head->spec.output_dim * sizeof(float));
    w->probabilities = malloc(head->spec.output_dim * sizeof(float));
    if (!w->input || !w->hidden || !w->logits || !w->probabilities) {
        decision_head_workspace_free(w); return NULL;
    }
    return w;
}

/* Four accumulators expose SIMD without fast-math, BLAS, or a thread pool. */
static float dot(const float *a, const float *b, size_t n) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    size_t j = 0;
    for (; j + 3 < n; j += 4) {
        s0 += a[j]*b[j]; s1 += a[j+1]*b[j+1];
        s2 += a[j+2]*b[j+2]; s3 += a[j+3]*b[j+3];
    }
    float s = (s0+s1)+(s2+s3);
    for (; j < n; j++) s += a[j]*b[j];
    return s;
}

int decision_head_predict(const struct decision_head *head, struct decision_head_workspace *w,
                          const float *input, size_t input_count, float *logits, float *probabilities,
                          size_t capacity, size_t *choice, char *error, size_t error_size) {
    if (choice) *choice = SIZE_MAX;
    if (error && error_size) error[0] = 0;
    if (!head || !w || w->owner != head || !input || !logits || !probabilities || logits == probabilities || !choice ||
        input_count != head->spec.features.input_dim || capacity < head->spec.output_dim)
        return fail(error, error_size, "Invalid decision-head inference buffers or workspace");
    const struct decision_head_spec *s = &head->spec;
    for (size_t j = 0; j < input_count; j++) {
        if (!isfinite(input[j])) return fail(error, error_size, "Nonfinite decision-head input");
        float v = (input[j]-s->mean[j])/s->scale[j];
        w->input[j] = v < -s->clip ? -s->clip : v > s->clip ? s->clip : v;
    }
    memset(w->logits, 0, s->output_dim * sizeof(float));
    for (size_t k = 0; k < s->ensemble_size; k++) {
        const struct decision_head_member *m = &s->members[k];
        const float *x = w->input;
        size_t width = input_count;
        if (s->hidden_dim) {
            for (size_t i = 0; i < s->hidden_dim; i++) {
                float v = dot(m->hidden_weight + i*input_count, x, input_count) + m->hidden_bias[i];
                if (!isfinite(v)) return fail(error, error_size, "Nonfinite hidden activation");
                w->hidden[i] = tanhf(v);
            }
            x = w->hidden; width = s->hidden_dim;
        }
        for (size_t i = 0; i < s->output_dim; i++)
            w->logits[i] += dot(m->output_weight + i*width, x, width) + m->output_bias[i];
    }
    size_t best = 0;
    for (size_t i = 0; i < s->output_dim; i++) {
        w->logits[i] /= (float)s->ensemble_size;
        w->probabilities[i] = w->logits[i]/(float)s->temperature;
        if (!isfinite(w->probabilities[i])) return fail(error, error_size, "Nonfinite decision logits");
        if (w->probabilities[i] > w->probabilities[best]) best = i;
    }
    float max = w->probabilities[best], total = 0;
    for (size_t i = 0; i < s->output_dim; i++) {
        w->probabilities[i] = expf(w->probabilities[i]-max);
        total += w->probabilities[i];
    }
    /* Match the reference's argmax of the FP32 probabilities, including ties
     * introduced by exp/normalization rounding between almost-equal logits. */
    best = 0;
    for (size_t i = 0; i < s->output_dim; i++) {
        w->probabilities[i] /= total;
        if (w->probabilities[i] > w->probabilities[best]) best = i;
    }
    memcpy(logits, w->logits, s->output_dim * sizeof(float));
    memcpy(probabilities, w->probabilities, s->output_dim * sizeof(float));
    *choice = best; return 0;
}
