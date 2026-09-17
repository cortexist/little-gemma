// Decode-first must be as safe as prefill-first: future wider activations
// must not invalidate buffers retained by a captured graph. No GGUF needed.
// cmake --build build --target cuda_scratch_test
// compute-sanitizer --tool memcheck --error-exitcode 1 build/cuda_scratch_test
#include "../src/cuda/model-cuda-i8.cu"
#include <vector>
int (*model_pick)(const float *, int) = NULL;
int g_mtp_n = LG_MTP_N;
int main(void) {
    model m{};
    int hd[] = {256}, nkv[] = {1};
    m.cfg.n_layer = 1; m.cfg.n_head = 4;
    m.cfg.n_embd = 256; m.cfg.n_ff = 512; m.cfg.n_vocab = 16;
    m.head_dim = hd; m.n_head_kv = nkv;
    ensure_scratch(&m);  // no preceding prefill; Q is wider than this model's FFN
    const int n = m.cfg.n_embd;
    std::vector<float> input(n);
    for (int i = 0; i < n; i++) input[i] = (i % 31 - 15) / 16.f;
    CUDA_CHECK(cudaMemcpy(dx, input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    actq small = actq_for(n);
    quantize_act_kernel<<<1, 256>>>(dx, small, n / 32);
    std::vector<unsigned char> expected(n), actual(n);
    std::vector<float2> expected_scales(n / 32), actual_scales(n / 32);
    CUDA_CHECK(cudaMemcpy(expected.data(), small.xq, n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(expected_scales.data(), small.xds, n / 32 * sizeof(float2), cudaMemcpyDeviceToHost));
    cudaGraph_t graph; cudaGraphExec_t exec;
    CUDA_CHECK(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    quantize_act_kernel<<<1, 256>>>(dx, small, n / 32);
    CUDA_CHECK(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    actq wide = actq_for(g_prefill_max_b * m.cfg.n_head * hd[0]);
    if (wide.xq != small.xq || wide.xds != small.xds) {
        fprintf(stderr, "prefill moved buffers retained by a decode graph\n"); return 2;
    }
    CUDA_CHECK(cudaMemset(small.xq, 0xff, n));
    CUDA_CHECK(cudaGraphLaunch(exec, cudaStreamPerThread));
    CUDA_CHECK(cudaMemcpy(actual.data(), small.xq, n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(actual_scales.data(), small.xds, n / 32 * sizeof(float2), cudaMemcpyDeviceToHost));
    if (actual != expected || memcmp(actual_scales.data(), expected_scales.data(), n / 32 * sizeof(float2))) {
        fprintf(stderr, "decode graph changed after wider activation request\n"); return 2;
    }
    CUDA_CHECK(cudaGraphExecDestroy(exec)); CUDA_CHECK(cudaGraphDestroy(graph));
    puts("CUDA scratch graph-lifetime check passed");
    return 0;
}
