#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "graph.h"

static int64_t n_elements(const struct tensor *t) {
    int64_t n = 1;
    for (int i = 0; i < t->n_dims; i++) n *= t->dims[i];
    return n;
}

// Allocate a node and append it to the graph. If alloc_data, also allocate its
// f32 result buffer (used by ops; leaves borrow their data instead).
static struct tensor *node_new(struct graph *g, const char *name,
                               int n_dims, const int64_t *dims, int alloc_data) {
    if (g->n_nodes == g->cap) {
        int cap = g->cap ? g->cap * 2 : 16;
        struct tensor **p = realloc(g->nodes, (size_t)cap * sizeof(*p));
        if (!p) return NULL;
        g->nodes = p;
        g->cap = cap;
    }

    struct tensor *t = calloc(1, sizeof(*t));
    if (!t) return NULL;
    snprintf(t->name, sizeof(t->name), "%s", name ? name : "");
    t->n_dims = n_dims;
    for (int i = 0; i < n_dims; i++) t->dims[i] = dims[i];

    if (alloc_data) {
        t->data = malloc((size_t)n_elements(t) * sizeof(float));
        if (!t->data) { free(t); return NULL; }
        t->owns_data = 1;
    }

    g->nodes[g->n_nodes++] = t;
    return t;
}

struct graph *graph_new(void) {
    return calloc(1, sizeof(struct graph));
}

void graph_free(struct graph *g) {
    if (!g) return;
    for (int i = 0; i < g->n_nodes; i++) {
        if (g->nodes[i]->owns_data) free(g->nodes[i]->data);
        free(g->nodes[i]);
    }
    free(g->nodes);
    free(g);
}

struct tensor *tensor_leaf(struct graph *g, const char *name,
                           int n_dims, const int64_t *dims, float *data) {
    struct tensor *t = node_new(g, name, n_dims, dims, 0);
    if (!t) return NULL;
    t->op = OP_NONE;
    t->data = data; // borrowed: points at a weight or an input
    return t;
}

// ---- op builders ----------------------------------------------------------
//
// Each builder is given tensors that already exist, so it can read their shapes
// to size its own result. Because the sources are always created first, the new
// node is appended after them -- which is why creation order is a valid order to
// run the graph in.

struct tensor *tensor_matmul(struct graph *g, struct tensor *w, struct tensor *x) {
    // w: (k, m) with dims[0]=k the contraction axis, dims[1]=m the output rows.
    // x: (k,)  ->  result: (m,). This matches GGUF weight layout (dims[0]=in).
    int64_t dims[1] = { w->dims[1] };
    struct tensor *t = node_new(g, "matmul", 1, dims, 1);
    if (!t) return NULL;
    t->op = OP_MATMUL;
    t->src[0] = w;
    t->src[1] = x;
    return t;
}

struct tensor *tensor_rmsnorm(struct graph *g, struct tensor *x, struct tensor *weight, float eps) {
    struct tensor *t = node_new(g, "rmsnorm", x->n_dims, x->dims, 1);
    if (!t) return NULL;
    t->op = OP_RMSNORM;
    t->src[0] = x;
    t->src[1] = weight;
    t->eps = eps;
    return t;
}

struct tensor *tensor_mul(struct graph *g, struct tensor *a, struct tensor *b) {
    struct tensor *t = node_new(g, "mul", a->n_dims, a->dims, 1);
    if (!t) return NULL;
    t->op = OP_MUL;
    t->src[0] = a;
    t->src[1] = b;
    return t;
}

struct tensor *tensor_add(struct graph *g, struct tensor *a, struct tensor *b) {
    struct tensor *t = node_new(g, "add", a->n_dims, a->dims, 1);
    if (!t) return NULL;
    t->op = OP_ADD;
    t->src[0] = a;
    t->src[1] = b;
    return t;
}

struct tensor *tensor_silu(struct graph *g, struct tensor *x) {
    struct tensor *t = node_new(g, "silu", x->n_dims, x->dims, 1);
    if (!t) return NULL;
    t->op = OP_SILU;
    t->src[0] = x;
    return t;
}

struct tensor *tensor_softmax(struct graph *g, struct tensor *x) {
    struct tensor *t = node_new(g, "softmax", x->n_dims, x->dims, 1);
    if (!t) return NULL;
    t->op = OP_SOFTMAX;
    t->src[0] = x;
    return t;
}

// ---- op kernels (plain f32, single-threaded; this is the part CUDA replaces) -

static void op_matmul(struct tensor *t) {
    const struct tensor *w = t->src[0];
    const struct tensor *x = t->src[1];
    int64_t k = w->dims[0];
    int64_t m = w->dims[1];
    for (int64_t i = 0; i < m; i++) {
        const float *row = w->data + i * k; // row i has k contiguous values
        float sum = 0.0f;
        for (int64_t j = 0; j < k; j++) sum += row[j] * x->data[j];
        t->data[i] = sum;
    }
}

static void op_rmsnorm(struct tensor *t) {
    const struct tensor *x = t->src[0];
    const struct tensor *w = t->src[1];
    int64_t n = n_elements(x);
    float ss = 0.0f;
    for (int64_t i = 0; i < n; i++) ss += x->data[i] * x->data[i];
    float scale = 1.0f / sqrtf(ss / (float)n + t->eps);
    for (int64_t i = 0; i < n; i++) t->data[i] = x->data[i] * scale * w->data[i];
}

static void op_mul(struct tensor *t) {
    int64_t n = n_elements(t);
    for (int64_t i = 0; i < n; i++) t->data[i] = t->src[0]->data[i] * t->src[1]->data[i];
}

static void op_add(struct tensor *t) {
    int64_t n = n_elements(t);
    for (int64_t i = 0; i < n; i++) t->data[i] = t->src[0]->data[i] + t->src[1]->data[i];
}

static void op_silu(struct tensor *t) {
    int64_t n = n_elements(t);
    for (int64_t i = 0; i < n; i++) {
        float v = t->src[0]->data[i];
        t->data[i] = v / (1.0f + expf(-v)); // v * sigmoid(v)
    }
}

static void op_softmax(struct tensor *t) {
    const float *x = t->src[0]->data;
    int64_t n = n_elements(t);
    float max = x[0];
    for (int64_t i = 1; i < n; i++) if (x[i] > max) max = x[i];
    float sum = 0.0f;
    for (int64_t i = 0; i < n; i++) { t->data[i] = expf(x[i] - max); sum += t->data[i]; }
    for (int64_t i = 0; i < n; i++) t->data[i] /= sum;
}

void graph_compute(struct graph *g) {
    // Sources always precede their results, so a single forward pass is correct.
    for (int i = 0; i < g->n_nodes; i++) {
        struct tensor *t = g->nodes[i];
        switch (t->op) {
            case OP_NONE:                    break; // leaf: data already present
            case OP_MATMUL:  op_matmul(t);   break;
            case OP_RMSNORM: op_rmsnorm(t);  break;
            case OP_MUL:     op_mul(t);      break;
            case OP_ADD:     op_add(t);      break;
            case OP_SILU:    op_silu(t);     break;
            case OP_SOFTMAX: op_softmax(t);  break;
        }
    }
}

const char *op_name(int op) {
    switch (op) {
        case OP_NONE:    return "leaf";
        case OP_MATMUL:  return "matmul";
        case OP_RMSNORM: return "rmsnorm";
        case OP_MUL:     return "mul";
        case OP_ADD:     return "add";
        case OP_SILU:    return "silu";
        case OP_SOFTMAX: return "softmax";
        default:         return "?";
    }
}

void graph_print(const struct graph *g) {
    printf("graph: %d nodes\n", g->n_nodes);
    for (int i = 0; i < g->n_nodes; i++) {
        const struct tensor *t = g->nodes[i];
        printf("  [%2d] %-8s %-12s dims=[", i, op_name(t->op), t->name);
        for (int d = 0; d < t->n_dims; d++) {
            if (d) printf(", ");
            printf("%lld", (long long)t->dims[d]);
        }
        printf("]");
        for (int s = 0; s < TENSOR_MAX_SRC && t->src[s]; s++) {
            printf(" %s %s", s ? "," : "<-", t->src[s]->name);
        }
        printf("\n");
    }
}
