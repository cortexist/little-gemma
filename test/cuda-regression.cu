// GPU-only regression gates for private CUDA helpers and KV allocation.
// Build explicitly: cmake --build build --target cuda_regression_test
// Run normally, then with LG_PREFILL_MAX_B=801 LG_WIDE_CHUNK=128,
// LG_SWA_F32=1, and LG_FLASH_REG=1 in separate processes. For memory checks:
// compute-sanitizer --tool memcheck --error-exitcode 1 build/cuda_regression_test
// Include the backend to exercise the production kernels without a test API.
#include "../src/cuda/model-cuda-i8.cu"
#include <vector>

int (*model_pick)(const float *, int) = NULL;
int g_mtp_n = LG_MTP_N;

__global__ static void half_bits(uint32_t *out) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    out[2 * i] = __float_as_uint(d_fp16((uint16_t)i));
    out[2 * i + 1] = __float_as_uint(d_bf16((uint16_t)i));
}

static void check_half(void) {
    std::vector<uint16_t> in(65536);
    std::vector<uint32_t> got(2 * in.size());
    std::vector<float> ref(in.size());
    for (unsigned i = 0; i < in.size(); i++) in[i] = (uint16_t)i;
    uint32_t *out;
    CUDA_CHECK(cudaMalloc(&out, got.size() * sizeof(uint32_t)));
    half_bits<<<256, 256>>>(out);
    CUDA_CHECK(cudaMemcpy(got.data(), out, got.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (!dequantize_into(GGML_TYPE_F16, in.data(), ref.data(), in.size())) exit(2);
    for (unsigned i = 0; i < in.size(); i++) {
        uint32_t bits; memcpy(&bits, &ref[i], sizeof bits);
        if (got[2 * i] != bits || got[2 * i + 1] != i << 16) {
            fprintf(stderr, "half conversion mismatch: 0x%04x\n", i); exit(2);
        }
    }
    CUDA_CHECK(cudaFree(out));
}

static void check_quant(void) {
    const int n = 32768;
    std::vector<float> x(n);
    uint32_t rng = 7;
    for (int i = 32; i < n; i++) {
        rng = rng * 1664525u + 1013904223u;
        x[i] = ldexpf((float)((int)(rng >> 8) - 0x800000), (i / 32) % 24 - 28);
    }
    float *dx;
    actq a{}, b{};
    CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    for (actq *q : {&a, &b}) {
        CUDA_CHECK(cudaMalloc(&q->xq, n));
        CUDA_CHECK(cudaMalloc(&q->xds, n / 32 * sizeof(float2)));
    }
    quantize_act_kernel<<<4, 256>>>(dx, a, n / 32);
    quantize_act_n_kernel<<<128, 256>>>(dx, b, n / 32);
    std::vector<unsigned char> left(n), right(n);
    for (int scales = 0; scales < 2; scales++) {
        size_t bytes = scales ? n / 32 * sizeof(float2) : n;
        CUDA_CHECK(cudaMemcpy(left.data(), scales ? (void *)a.xds : a.xq, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(right.data(), scales ? (void *)b.xds : b.xq, bytes, cudaMemcpyDeviceToHost));
        if (memcmp(left.data(), right.data(), bytes)) { fprintf(stderr, "activation quantization mismatch\n"); exit(2); }
    }
    for (actq *q : {&a, &b}) { CUDA_CHECK(cudaFree(q->xq)); CUDA_CHECK(cudaFree(q->xds)); }
    CUDA_CHECK(cudaFree(dx));
}

static void check_cache(void) {
    model m{};
    int local[] = {0, 1}, hd[] = {512, 256}, nkv[] = {1, 1};
    m.cfg.n_layer = m.cfg.n_kv_start = m.cfg.n_head = 2;
    m.cfg.sliding_window = 8;
    m.is_local = local; m.head_dim = hd; m.n_head_kv = nkv;
    model_prefill_reserve();
    if (g_prefill_max_b % 32) { fprintf(stderr, "unaligned media reserve\n"); exit(2); }
    for (int capacity : {33, 63, 65, 2049}) {
        kvcache kv{};
        if (kvcache_init(&kv, &m, capacity)) exit(2);
        if (kv.max_seq != capacity || kv.seq[0] != capacity) exit(2);
        int pos = capacity - 32, *dp;
        CUDA_CHECK(cudaMalloc(&dp, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(dp, &pos, sizeof(int), cudaMemcpyHostToDevice));
        for (int l = 0; l < 2; l++) {
            size_t bytes = (size_t)32 * m.cfg.n_head * hd[l] * sizeof(float);
            float *q, *out;
            CUDA_CHECK(cudaMalloc(&q, bytes)); CUDA_CHECK(cudaMemset(q, 0, bytes));
            CUDA_CHECK(cudaMalloc(&out, bytes)); CUDA_CHECK(cudaMemset(out, 0xff, bytes));
            bool ring = kv.seq[l] < kv.max_seq;
            if (hd[l] == 512)
                launch_flash<512>(out, q, kv.k[l], kv.v[l], hd[l], 2, dp, 0, kv.seq[l], 32, 2, kv.f16[l], ring, NULL, 0);
            else
                launch_flash<256>(out, q, kv.k[l], kv.v[l], hd[l], 2, dp, 8, kv.seq[l], 32, 2, kv.f16[l], ring, NULL, 0);
            std::vector<float> result(bytes / sizeof(float));
            CUDA_CHECK(cudaMemcpy(result.data(), out, bytes, cudaMemcpyDeviceToHost));
            for (float v : result) if (v != 0.0f) { fprintf(stderr, "flash tail mismatch\n"); exit(2); }
            CUDA_CHECK(cudaFree(q)); CUDA_CHECK(cudaFree(out));
        }
        CUDA_CHECK(cudaFree(dp)); kvcache_free(&kv);
    }
}

int main(void) {
    check_half(); check_quant(); check_cache();
    CUDA_CHECK(cudaDeviceSynchronize());
    puts("CUDA regression checks passed");
}
