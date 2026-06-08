#include <stdio.h>
#include <math.h>

#include "graph.h"

static int failures = 0;

static void check(const char *what, float got, float want) {
    int ok = fabsf(got - want) < 1e-4f;
    printf("  %-28s got %9.5f  want %9.5f  %s\n", what, got, want, ok ? "ok" : "FAIL");
    if (!ok) failures++;
}

int main(void) {
    struct graph *g = graph_new();

    // Inputs.
    float xv[3] = { 1.0f, 2.0f, 3.0f };
    float onesv[3] = { 1.0f, 1.0f, 1.0f };
    int64_t d3[1] = { 3 };
    struct tensor *x    = tensor_leaf(g, "x",    1, d3, xv);
    struct tensor *ones = tensor_leaf(g, "ones", 1, d3, onesv);

    // A (k=3, m=2) weight selecting the first two components: rows {1,0,0},{0,1,0}.
    float wv[6] = { 1, 0, 0,   0, 1, 0 };
    int64_t wdims[2] = { 3, 2 };
    struct tensor *w = tensor_leaf(g, "w", 2, wdims, wv);

    struct tensor *mm   = tensor_matmul(g, w, x);          // -> [1, 2]
    struct tensor *rms  = tensor_rmsnorm(g, x, ones, 0.0f); // normalize, scale by 1
    struct tensor *add  = tensor_add(g, x, x);             // -> [2, 4, 6]
    struct tensor *mul  = tensor_mul(g, x, x);             // -> [1, 4, 9]
    struct tensor *si   = tensor_silu(g, x);              // silu([1,2,3])
    struct tensor *sm   = tensor_softmax(g, x);           // softmax([1,2,3])

    graph_print(g);
    graph_compute(g);

    printf("\nchecks:\n");
    check("matmul[0]",  mm->data[0], 1.0f);
    check("matmul[1]",  mm->data[1], 2.0f);

    float rms_scale = 1.0f / sqrtf((1 + 4 + 9) / 3.0f);
    check("rmsnorm[2]", rms->data[2], 3.0f * rms_scale);

    check("add[2]",     add->data[2], 6.0f);
    check("mul[2]",     mul->data[2], 9.0f);
    check("silu[1]",    si->data[1], 2.0f / (1.0f + expf(-2.0f)));

    float e1 = expf(1 - 3.0f), e2 = expf(2 - 3.0f), e3 = expf(3 - 3.0f);
    check("softmax[2]", sm->data[2], e3 / (e1 + e2 + e3));

    graph_free(g);

    printf("\n%s\n", failures ? "FAILED" : "all checks passed");
    return failures ? 1 : 0;
}
