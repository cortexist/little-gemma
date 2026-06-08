#ifndef GRAPH_H
#define GRAPH_H

#include <stdint.h>

// A small computation graph: tensors are the nodes, and each non-leaf tensor
// remembers the op that produced it and the tensors it came from. This is the
// stripped-to-the-bone version of what ggml does, so you can see what a tensor
// and a graph really are.
//
// Values are plain f32 in this layer. Weights stored quantized in the GGUF file
// are expected to be dequantized to f32 before being wrapped as a leaf tensor.

enum op {
    OP_NONE = 0,   // a leaf: a weight or an input, nothing to compute
    OP_MATMUL,     // w (k,m) times x (k,) -> (m,)
    OP_RMSNORM,    // normalize x, then scale by a weight vector
    OP_MUL,        // elementwise a * b
    OP_ADD,        // elementwise a + b (residual connections)
    OP_SILU,       // SiLU / swish activation: x * sigmoid(x)
    OP_SOFTMAX,    // softmax over all elements
};

#define TENSOR_MAX_DIMS 4
#define TENSOR_MAX_SRC  2

struct tensor {
    char           name[64];
    int            type;                  // ggml type tag; this layer computes in f32
    int            n_dims;
    int64_t        dims[TENSOR_MAX_DIMS]; // dims[0] is the innermost (fastest) axis
    float         *data;                  // leaf: borrowed; result: owned by the graph
    int            op;                     // OP_NONE for a leaf
    struct tensor *src[TENSOR_MAX_SRC];   // the inputs this op reads
    float          eps;                    // op parameter (used by OP_RMSNORM)
    int            owns_data;              // graph frees data on teardown if set
};

struct graph {
    struct tensor **nodes;   // every tensor, in creation == execution order
    int             n_nodes;
    int             cap;
};

// lifecycle
struct graph *graph_new(void);
void          graph_free(struct graph *g);

// A leaf holds values the caller already has (e.g. a weight pointing at the
// dequantized GGUF data, or the input activation). The data is borrowed, not freed.
struct tensor *tensor_leaf(struct graph *g, const char *name,
                           int n_dims, const int64_t *dims, float *data);

// Ops. Each allocates its result (shape inferred from the inputs), records the
// op and its sources, appends the node to the graph, and returns it.
struct tensor *tensor_matmul (struct graph *g, struct tensor *w, struct tensor *x);
struct tensor *tensor_rmsnorm(struct graph *g, struct tensor *x, struct tensor *weight, float eps);
struct tensor *tensor_mul    (struct graph *g, struct tensor *a, struct tensor *b);
struct tensor *tensor_add    (struct graph *g, struct tensor *a, struct tensor *b);
struct tensor *tensor_silu   (struct graph *g, struct tensor *x);
struct tensor *tensor_softmax(struct graph *g, struct tensor *x);

// Run every node in order. Each result ends up in its own ->data.
void graph_compute(struct graph *g);

// Print the graph (index, op, shape, and the sources each node reads).
void graph_print(const struct graph *g);

// Human-readable name for an op code.
const char *op_name(int op);

#endif // GRAPH_H
