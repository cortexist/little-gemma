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

// Compare grouped dispatch to separate tiles, including narrow and wide tails.
static void check_mma_tails(void) {
    const int K = 768, M = 259, B = 1120;
    std::vector<float> x(K * B);
    uint32_t rng = 17;
    for (float &v : x) { rng = rng * 1664525u + 1013904223u; v = ((int)(rng >> 8) - 0x800000) / 8388608.f; }
    float *xd, *y;
    CUDA_CHECK(cudaMalloc(&xd, x.size() * 4)); CUDA_CHECK(cudaMalloc(&y, M * B * 4));
    CUDA_CHECK(cudaMemcpy(xd, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    actq aq = actq_for(K * B);
    quantize_act_n_kernel<<<(K * B / 32 + 7) / 8, 256>>>(xd, aq, K * B / 32);
    int types[] = {GGML_TYPE_Q4_K, GGML_TYPE_Q4_0, GGML_TYPE_Q6_K};
    for (int type : types) {
        int bl = ggml_blck_size(type), ts = ggml_type_size(type), nb = K * M / bl;
        std::vector<unsigned char> raw((size_t)nb * ts);
        for (auto &v : raw) { rng = rng * 1664525u + 1013904223u; v = rng >> 24; }
        for (int i = 0; i < nb; i++) {
            void *p = raw.data() + (size_t)i * ts;
            uint16_t h = __half_as_ushort(__float2half((i % 29 + 1) / 512.f));
            switch (type) {
                case GGML_TYPE_Q4_K: ((block_q4_K *)p)->d = h; ((block_q4_K *)p)->dmin = h; break;
                case GGML_TYPE_Q6_K: ((block_q6_K *)p)->d = h; break;
                case GGML_TYPE_Q4_0: ((block_q4_0 *)p)->d = h; break;
            }
        }
        std::vector<unsigned char> rep;
        if (type == GGML_TYPE_Q6_K) {
            ts = sizeof(block_q6_Kr); rep.resize((size_t)nb * ts);
            repack_q6_K((block_q6_Kr *)rep.data(), (block_q6_K *)raw.data(), nb);
        } else if (type == GGML_TYPE_Q4_0) {
            nb /= 8; ts = sizeof(block_q4_0m); rep.resize((size_t)nb * ts);
            repack_q4_0m((block_q4_0m *)rep.data(), (block_q4_0 *)raw.data(), nb);
        }
        auto &host = rep.empty() ? raw : rep;
        unsigned char *w;
        CUDA_CHECK(cudaMalloc(&w, host.size())); CUDA_CHECK(cudaMemcpy(w, host.data(), host.size(), cudaMemcpyHostToDevice));

        cudaDeviceProp props; CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
        int sms = props.multiProcessorCount;
        for (int cols : {32, 64, 96, 128, 160, 192, 288, 1120}) {
            g_pf_cols = cols;
            if (type == GGML_TYPE_Q6_K) matmul_q6_chunk(y, w, ts, K, M, sms);
            else if (type == GGML_TYPE_Q4_K) matmul_q4_chunk<0>(y, (block_q4_K *)w, K, M, sms);
            else matmul_q4_chunk<1>(y, (block_q4_K *)w, K, M, sms);
            std::vector<float> got(M * cols), ref(M * cols);
            CUDA_CHECK(cudaMemcpy(got.data(), y, got.size() * 4, cudaMemcpyDeviceToHost));
            for (int c = 0; c < cols; ) {
                #define TILE(C) do { \
                    float *out = y + (size_t)c * M; \
                    const int8_t *xq = aq.xq + (size_t)c * K; \
                    const float2 *ds = aq.xds + (size_t)c * (K / 32); \
                    if (type == GGML_TYPE_Q6_K) launch_q6k_mmq<C>(out, w, ts, xq, ds, K, M, sms); \
                    else if (type == GGML_TYPE_Q4_K) launch_q4k_mma<C, 0>(out, (block_q4_K *)w, xq, ds, K, M, sms); \
                    else launch_q4k_mma<C, 1>(out, (block_q4_K *)w, xq, ds, K, M, sms); \
                } while (0)
                if (cols - c >= 64) { TILE(64); c += 64; }
                else { TILE(32); c += 32; }
                #undef TILE
            }
            CUDA_CHECK(cudaMemcpy(ref.data(), y, ref.size() * 4, cudaMemcpyDeviceToHost));
            if (memcmp(got.data(), ref.data(), got.size() * 4)) {
                fprintf(stderr, "MMA tail mismatch: type=%d cols=%d\n", type, cols); exit(2);
            }
        }
        CUDA_CHECK(cudaFree(w));
    }
    g_pf_cols = 0;
    CUDA_CHECK(cudaFree(xd)); CUDA_CHECK(cudaFree(y));
}

static void check_argmax(void) {
    for (int n : {1, 31, 1025, 262144}) {
        std::vector<float> x((size_t)n * LG_MTP_N_MAX);
        for (int j = 0; j < LG_MTP_N_MAX; j++)
            for (int i = 0; i < n; i++) x[(size_t)j * n + i] = (i + j) % 17 - 8.f;
        float *d; int *a, *b;
        CUDA_CHECK(cudaMalloc(&d, x.size() * 4));
        CUDA_CHECK(cudaMemcpy(d, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&a, LG_MTP_N_MAX * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&b, LG_MTP_N_MAX * sizeof(int)));
        for (int count = 2; count <= LG_MTP_N_MAX; count++) {
            for (int j = 0; j < count; j++) argmax_kernel<><<<1, 1024>>>(d + (size_t)j * n, n, a + j);
            argmax_kernel<true><<<count, 1024>>>(d, n, b);
            int ref[LG_MTP_N_MAX], got[LG_MTP_N_MAX];
            CUDA_CHECK(cudaMemcpy(ref, a, count * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(got, b, count * sizeof(int), cudaMemcpyDeviceToHost));
            if (memcmp(ref, got, count * sizeof(int))) { fprintf(stderr, "batched argmax mismatch\n"); exit(2); }
        }
        CUDA_CHECK(cudaFree(d)); CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(b));
    }
}

int main(void) {
    check_half(); check_quant(); check_cache(); check_mma_tails(); check_argmax();
    CUDA_CHECK(cudaDeviceSynchronize());
    puts("CUDA regression checks passed");
}
