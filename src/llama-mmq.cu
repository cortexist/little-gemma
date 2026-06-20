// ============================================================================
// [0] PROVENANCE
// ============================================================================
// llama-mmq.cu — llama.cpp's q4_K MMQ matmul, VENDORED, PRUNED and DE-BAGGAGED
// into one self-contained CUDA translation unit. Replaces the flash-llama.cu
// bridge's external -I dependency on a pinned llama checkout: the device code
// below is llama's REAL code, pruned at whole-function / whole-struct /
// whole-overload granularity to the q4_K path. The only hand-written code is
// the "ggml shim" in section [1] (which replaces common.cuh + ggml-common.h +
// ggml.h, holding ONLY the type structs, enums, macros/constants, and helpers
// the q4_K path needs — most of them themselves copied out of the ggml
// headers), and the bridge + ggml stubs in section [5] (carried over from
// flash-llama.cu).
//
// Source: llama.cpp fork at commit 5fcc6454da2e35c0963855344ce17311c2f1ff0f
//   ggml/src/ggml-cuda/{mmq.cuh, mma.cuh, common.cuh, vecdotq.cuh, quantize.cu}
//   ggml/src/ggml-common.h, ggml/include/ggml.h
//
// A .cu file is NVIDIA CUDA ONLY (an AMD port would be a separate Vulkan compute
// shader, never sharing these sources). So this is no longer a verbatim copy:
// the non-NVIDIA backend scaffolding (HIP/AMD/MUSA/MThreads — RDNA*/CDNA/GCN/
// MFMA/WMMA gates, predicate macros, helpers) and the LEGACY Volta (sm_70) arch
// arms have been removed, collapsing each #if to the arm that is live on the
// supported NVIDIA target. "Cut off the past, don't cut off the future": the
// forward-looking Blackwell (sm_100+) branches are KEPT intact even though they
// are dead on sm_86. Behaviorally BIT-IDENTICAL to the pristine vendor copy on
// the live sm_86 path (verified via memcmp == 0 across stream-K / need_check /
// fixup shapes); line-range citations on each section are retained.
//
// Builds standalone for sm_86 (RTX A5000):
//   nvcc -c -arch=sm_86 -I<little-gemma/include> --extended-lambda src/llama-mmq.cu
// No -I into the llama checkout, no -DGGML_USE_CUDA: the shim is self-contained.
//
// One deliberate post-vendor edit: llama's kernels were renamed to fit our naming
// (mul_mat_q -> matmul_q_prefill, mul_mat_q_process_tile -> matmul_q_prefill_process_tile,
// mul_mat_q_stream_k_fixup -> matmul_q_prefill_stream_k_fixup). Names only; bodies verbatim.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>   // nv_bfloat162 (mma.cuh's bf16 tile<>/mma() specializations)
#include <cstdint>
#include <cstdlib>
#include <climits>
#include <cstdio>
#include <type_traits>
#include <limits>

// ============================================================================
// [1] GGML SHIM  (HAND-WRITTEN — replaces common.cuh + ggml-common.h + ggml.h)
// ============================================================================
// Every macro/struct/enum/helper below that the kept llama device code needs.
// Definitions tagged "[verbatim ...]" are copied EXACTLY from the cited ggml
// header line range; the rest (tagged "[shim]") is the minimal hand-written
// scaffolding (stub macros for host-only ggml plumbing the kernel never reaches).
// This is a pure NVIDIA CUDA build: the non-NVIDIA backend gates (GGML_USE_HIP /
// GGML_USE_MUSA / AMD / MThreads) and the legacy Volta arms that the original
// llama code wraps in #ifs have been collapsed here to the NVIDIA arm, so there
// are no GGML_USE_* feature flags to force OFF anymore.

// --- [verbatim ggml.h:258] ---
#define GGML_UNUSED(x) (void)(x)
// --- [verbatim ggml.h:264] (non-MSVC arm; nvcc host is the relevant one) ---
#define GGML_UNUSED_VARS(...) do { (void)sizeof((__VA_ARGS__, 0)); } while(0)
// --- [verbatim ggml.h:267] ---
#define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))
// --- [verbatim ggml.h:287-288] ---
#define GGML_ABORT(...) ggml_abort(__FILE__, __LINE__, __VA_ARGS__)
#define GGML_ASSERT(x) if (!(x)) GGML_ABORT("GGML_ASSERT(%s) failed", #x)
// ggml_abort host symbol (stubbed in section [5]; matches ggml.h:355-356 signature,
// minus the GGML_NORETURN/format attributes — they aren't needed for the call sites
// and the GNU-attribute spelling isn't portable to the MSVC host compiler).
extern "C" void ggml_abort(const char * file, int line, const char * fmt, ...);

// --- [verbatim ggml.h:389-438]  enum ggml_type (full — the kept switch-helpers
//     reference many of its values as constexpr case labels) ---
    enum ggml_type {
        GGML_TYPE_F32     = 0,
        GGML_TYPE_F16     = 1,
        GGML_TYPE_Q4_0    = 2,
        GGML_TYPE_Q4_1    = 3,
        // GGML_TYPE_Q4_2 = 4, support has been removed
        // GGML_TYPE_Q4_3 = 5, support has been removed
        GGML_TYPE_Q5_0    = 6,
        GGML_TYPE_Q5_1    = 7,
        GGML_TYPE_Q8_0    = 8,
        GGML_TYPE_Q8_1    = 9,
        GGML_TYPE_Q2_K    = 10,
        GGML_TYPE_Q3_K    = 11,
        GGML_TYPE_Q4_K    = 12,
        GGML_TYPE_Q5_K    = 13,
        GGML_TYPE_Q6_K    = 14,
        GGML_TYPE_Q8_K    = 15,
        GGML_TYPE_IQ2_XXS = 16,
        GGML_TYPE_IQ2_XS  = 17,
        GGML_TYPE_IQ3_XXS = 18,
        GGML_TYPE_IQ1_S   = 19,
        GGML_TYPE_IQ4_NL  = 20,
        GGML_TYPE_IQ3_S   = 21,
        GGML_TYPE_IQ2_S   = 22,
        GGML_TYPE_IQ4_XS  = 23,
        GGML_TYPE_I8      = 24,
        GGML_TYPE_I16     = 25,
        GGML_TYPE_I32     = 26,
        GGML_TYPE_I64     = 27,
        GGML_TYPE_F64     = 28,
        GGML_TYPE_IQ1_M   = 29,
        GGML_TYPE_BF16    = 30,
        // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
        // GGML_TYPE_Q4_0_4_8 = 32,
        // GGML_TYPE_Q4_0_8_8 = 33,
        GGML_TYPE_TQ1_0   = 34,
        GGML_TYPE_TQ2_0   = 35,
        // GGML_TYPE_IQ4_NL_4_4 = 36,
        // GGML_TYPE_IQ4_NL_4_8 = 37,
        // GGML_TYPE_IQ4_NL_8_8 = 38,
        GGML_TYPE_MXFP4   = 39, // MXFP4 (1 block)
        GGML_TYPE_NVFP4   = 40, // NVFP4 (4 blocks, E4M3 scale)
        GGML_TYPE_Q1_0    = 41,
        GGML_TYPE_TURBO2_0 = 42, // TurboQuant 2-bit KV cache: WHT + 2-bit PolarQuant
        GGML_TYPE_TURBO3_0 = 43, // TurboQuant 3-bit KV cache: WHT + 3-bit PolarQuant
        GGML_TYPE_TURBO4_0 = 44, // TurboQuant 4-bit KV cache: WHT + 4-bit PolarQuant
        GGML_TYPE_TQ3_1S  = 45, // TurboQuant 3-bit weight: WHT-rotated 8-level Lloyd-Max, block_size=32
        GGML_TYPE_TQ4_1S  = 46, // TurboQuant 4-bit weight: WHT-rotated 16-level Lloyd-Max, block_size=32
        GGML_TYPE_COUNT   = 47,
    };

// --- [verbatim ggml-common.h:43-44] (CUDA arm: ggml_half==half) ---
typedef half  ggml_half;
typedef half2 ggml_half2;

// --- [verbatim ggml-common.h] block-quant constants. QK_K / K_SCALE_SIZE
//     (89-90), then the QR/QI integer-per-block constants (96-167) and the
//     QK* block-size defines (177,184,191,204,211-212,219,227,241,248,528).
//     The kept switch-helpers reference many of these as constexpr, so they
//     must all exist; copied verbatim. ---
#define QK_K 256
#define K_SCALE_SIZE 12

#define QI1_0 (QK1_0 / 32)
#define QR1_0 1


#define QI4_0 (QK4_0 / (4 * QR4_0))
#define QR4_0 2

#define QI4_1 (QK4_1 / (4 * QR4_1))
#define QR4_1 2

#define QI_MXFP4 (QK_MXFP4 / (4 * QR_MXFP4))
#define QR_MXFP4 2

#define QI_NVFP4 (QK_NVFP4 / (4 * QR_NVFP4))
#define QR_NVFP4 2

#define QI5_0 (QK5_0 / (4 * QR5_0))
#define QR5_0 2

#define QI5_1 (QK5_1 / (4 * QR5_1))
#define QR5_1 2

#define QI8_0 (QK8_0 / (4 * QR8_0))
#define QR8_0 1

#define QI8_1 (QK8_1 / (4 * QR8_1))
#define QR8_1 1

#define QI2_K (QK_K / (4*QR2_K))
#define QR2_K 4

#define QI3_K (QK_K / (4*QR3_K))
#define QR3_K 4

#define QI4_K (QK_K / (4*QR4_K))
#define QR4_K 2

#define QI5_K (QK_K / (4*QR5_K))
#define QR5_K 2

#define QI6_K (QK_K / (4*QR6_K))
#define QR6_K 2

#define QI2_XXS (QK_K / (4*QR2_XXS))
#define QR2_XXS 4

#define QI2_XS (QK_K / (4*QR2_XS))
#define QR2_XS 4

#define QI2_S (QK_K / (4*QR2_S))
#define QR2_S 4

#define QI3_XXS (QK_K / (4*QR3_XXS))
#define QR3_XXS 4

#define QI3_XS (QK_K / (4*QR3_XS))
#define QR3_XS 4

#define QI1_S (QK_K / (4*QR1_S))
#define QR1_S 8

#define QI1_M (QK_K / (4*QR1_M))
#define QR1_M 8

#define QI4_NL (QK4_NL / (4*QR4_NL))
#define QR4_NL 2

#define QI4_XS (QK_K / (4*QR4_XS))
#define QR4_XS 2

#define QI3_S (QK_K / (4*QR3_S))
#define QR3_S 4

#define QK1_0 128
#define QK4_0 32
#define QK4_1 32
#define QK_MXFP4 32
#define QK_NVFP4 64
#define QK_NVFP4_SUB 16  // sub-block size for per-group scales
#define QK5_0 32
#define QK5_1 32
#define QK8_0 32
#define QK8_1 32
#define QK4_NL 32

// --- [shim] GGML_EXTENSION / GGML_COMMON_AGGR_*: ggml-common.h selects these
//     per language/compiler. For the CUDA (nvcc, C++) arm they are
//     ggml-common.h:46-47 (GGML_COMMON_AGGR_U empty, _S = data) and 171-175
//     (GGML_EXTENSION = __extension__ on non-MSVC, empty on MSVC). nvcc's host
//     compiler is MSVC here, so use the MSVC arm verbatim. ---
#ifdef _MSC_VER
#define GGML_EXTENSION
#else
#define GGML_EXTENSION __extension__
#endif
#define GGML_COMMON_AGGR_U
#define GGML_COMMON_AGGR_S data

// --- [verbatim ggml-common.h:248-259]  block_q8_1 (q8_1 quantizer output) ---
typedef struct {
    GGML_EXTENSION union {
        struct {
            ggml_half d; // delta
            ggml_half s; // d * sum(qs[i])
        } GGML_COMMON_AGGR_S;
        ggml_half2 ds;
    } GGML_COMMON_AGGR_U;
    int8_t qs[QK8_1]; // quants
} block_q8_1;
static_assert(sizeof(block_q8_1) == 2*sizeof(ggml_half) + QK8_1, "wrong q8_1 block size/padding");

// --- [verbatim ggml-common.h:406-419]  block_q4_K (the weight layout) ---
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight
typedef struct {
    GGML_EXTENSION union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2, "wrong q4_K block size/padding");

// --- common.cuh symbols ---

// --- [verbatim common.cuh:40-43] ---
#define STRINGIZE_IMPL(...) #__VA_ARGS__
#define STRINGIZE(...) STRINGIZE_IMPL(__VA_ARGS__)

#define WARP_SIZE 32

// --- [verbatim common.cuh:47-60]  compute-capability constants the host
//     switch-helpers compare against (NVIDIA-only subset; AMD/MThreads CC
//     defines are pruned along with their predicates) ---
#define GGML_CUDA_CC_PASCAL          600
#define GGML_CUDA_CC_DP4A            610 // minimum compute capability for __dp4a, an intrinsic for byte-wise dot products
#define GGML_CUDA_CC_VOLTA           700
#define GGML_CUDA_CC_TURING          750
#define GGML_CUDA_CC_AMPERE          800
#define GGML_CUDA_CC_ADA_LOVELACE    890
#define GGML_CUDA_CC_BLACKWELL       1200
#define GGML_CUDA_CC_DGX_SPARK       1210
#define GGML_CUDA_CC_RUBIN           1300
#define GGML_CUDA_CC_OFFSET_MTHREADS 0x0100000
#define GGML_CUDA_CC_IS_NVIDIA(cc)   (cc < GGML_CUDA_CC_OFFSET_MTHREADS)

// --- [verbatim common.cuh:110-147]  __CUDA_ARCH_LIST__ machinery for
//     ggml_cuda_highest_compiled_arch (used by turing_mma_available and the
//     host switch-helpers). Kept verbatim so the arch selection matches. ---
#ifdef __CUDA_ARCH_LIST__
constexpr bool ggml_cuda_has_arch_impl(int) {
    return false;
}

template<class ... Archs>
constexpr bool ggml_cuda_has_arch_impl(const int arch, const int first, Archs... rest) {
    return arch == first || ggml_cuda_has_arch_impl(arch, rest...);
}

constexpr bool ggml_cuda_has_arch(const int arch) {
    return ggml_cuda_has_arch_impl(arch, __CUDA_ARCH_LIST__);
}

constexpr int ggml_cuda_highest_compiled_arch_impl(const int /*arch*/, const int cur) {
    if (cur == 0) {
        return -1;
    }
    return cur;
}

template<class ... Archs>
constexpr int ggml_cuda_highest_compiled_arch_impl(const int arch, const int cur, const int first, Archs... rest) {
    if (first <= arch && first > cur) {
        return ggml_cuda_highest_compiled_arch_impl(arch, first, rest...);
    } else {
        return ggml_cuda_highest_compiled_arch_impl(arch, cur, rest...);
    }
}

constexpr int ggml_cuda_highest_compiled_arch(const int arch) {
    return ggml_cuda_highest_compiled_arch_impl(arch, 0, __CUDA_ARCH_LIST__);
}
#else
static int ggml_cuda_highest_compiled_arch(const int arch) {
    return arch;
}
#endif // __CUDA_ARCH_LIST__

// --- [verbatim common.cuh:151] ---
#define MATRIX_ROW_PADDING 512 // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses

// --- [verbatim common.cuh:221-225, MUSA arm dropped] ---
#if CUDART_VERSION >= 11010
#define GGML_CUDA_ASSUME(x) __builtin_assume(x)
#else
#define GGML_CUDA_ASSUME(x)
#endif // CUDART_VERSION >= 11010

// --- [common.cuh:231-237, HIP/MUSA arms dropped] FP16 availability (drives
//     FP16_AVAILABLE / FAST_FP16_AVAILABLE used by ggml_cuda_mad/warp_reduce) ---
#if __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
#define FP16_AVAILABLE
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL

#if defined(FP16_AVAILABLE) && __CUDA_ARCH__ != 610
#define FAST_FP16_AVAILABLE
#endif // defined(FP16_AVAILABLE) && __CUDA_ARCH__ != 610

// --- [common.cuh:247-274, HIP arms dropped, legacy Volta removed]  the
//     MMA-availability arch gates. On sm_86 (NVIDIA, >= TURING and >= AMPERE)
//     TURING_MMA_AVAILABLE, AMPERE_MMA_AVAILABLE, CP_ASYNC_AVAILABLE,
//     LDMATRIX_TRANS_AVAILABLE are defined; BLACKWELL_MMA_AVAILABLE is dead on
//     sm_86 but KEPT for forward (sm_100+) support. ---
#if __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
#define TURING_MMA_AVAILABLE
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_TURING

#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#define AMPERE_MMA_AVAILABLE
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE

#if __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL && __CUDA_ARCH__ < GGML_CUDA_CC_RUBIN
#    define BLACKWELL_MMA_AVAILABLE
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL

#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#define CP_ASYNC_AVAILABLE
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE

#if defined(TURING_MMA_AVAILABLE)
#define LDMATRIX_TRANS_AVAILABLE
#endif // defined(TURING_MMA_AVAILABLE)

// --- [common.cuh:310-328, AMD helpers removed]  the arch-predicate host helper
//     the kept host switch-helpers call (turing_mma_available is the real body;
//     amd_mfma_available/amd_wmma_available were NVIDIA-always-false and their
//     callers below are simplified to the NVIDIA path). ---
static bool turing_mma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING;
}

// --- [common.cuh:343-362, HIP/GFX arms dropped] ---
static constexpr __device__ int ggml_cuda_get_physical_warp_size() {
    return 32;
}

// Maximum number of bytes that can be copied in a single instruction.
// [common.cuh:343-362, HIP arm dropped + legacy (< Volta, returned 8) removed]
static constexpr __device__ int ggml_cuda_get_max_cpy_bytes() {
    return 16;
}

// --- [common.cuh:365-390, HIP/MUSA arms dropped]  no_device_code + NO_DEVICE_CODE ---
[[noreturn]]
static __device__ void no_device_code(
    const char * file_name, const int line, const char * function_name, const int arch, const char * arch_list) {

    printf("%s:%d: ERROR: CUDA kernel %s has no device code compatible with CUDA arch %d. ggml-cuda.cu was compiled for: %s\n",
           file_name, line, function_name, arch, arch_list);
    __trap();

    GGML_UNUSED(no_device_code); // suppress unused function warning
}

#ifdef __CUDA_ARCH__
#define NO_DEVICE_CODE no_device_code(__FILE__, __LINE__, __FUNCTION__, __CUDA_ARCH__, STRINGIZE(__CUDA_ARCH_LIST__))
#else
#define NO_DEVICE_CODE //GGML_ABORT("NO_DEVICE_CODE not valid in host code.")
#endif // __CUDA_ARCH__

// [common.cuh:672-710, REMOVED]  ggml_cuda_dp4a was the byte-wise dot intrinsic
// the dp4a q4_K dot used; tensor-core-only, so it is unreferenced.

// --- [verbatim common.cuh:759-788]  ggml_cuda_memcpy_1 (used by mma.cuh
//     load paths) ---
template <int nbytes, int alignment = 0>
static __device__ __forceinline__ void ggml_cuda_memcpy_1(void * __restrict__ dst, const void * __restrict__ src) {
    static_assert(
        nbytes <= ggml_cuda_get_max_cpy_bytes() || alignment == 0,
        "You are misusing the alignment parameter for ggml_cuda_memcpy_1. "
        "The intent is for the parameter is only as a workaround if either one of the pointers is not properly aligned. "
        "If you use it to do more bytes per copy than ggml_cuda_max_cpy_bytes() the reads and writes may not be coalesced. "
        "Call ggml_cuda_memcpy_1 in a loop instead.");
    if constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    for (int i = 0; i < nbytes/nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1) {
            ((char *) dst)[i] = ((const char *) src)[i];
        } else if constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((const short *) src)[i];
        } else if constexpr (nb_per_cpy == 4) {
            ((int *) dst)[i] = ((const int *) src)[i];
        } else if constexpr (nb_per_cpy == 8) {
            ((int2 *) dst)[i] = ((const int2 *) src)[i];
        } else if constexpr (nb_per_cpy == 16) {
            ((int4 *) dst)[i] = ((const int4 *) src)[i];
        } else {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}

// --- [verbatim common.cuh:918-1002 subset]  ggml_cuda_type_traits primary
//     template + the GGML_TYPE_Q4_K specialization (the only one instantiated:
//     matmul_q_prefill reads ::qk). qk=QK_K, qr=QR4_K, qi=QI4_K. ---
template <ggml_type type>
struct ggml_cuda_type_traits;

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q4_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_K;
    static constexpr int qi = QI4_K;
};

// --- [verbatim quantize.cuh:8-9] ---
#define CUDA_QUANTIZE_BLOCK_SIZE     256
#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

static_assert(MATRIX_ROW_PADDING %    CUDA_QUANTIZE_BLOCK_SIZE      == 0, "Risk of out-of-bounds access.");
static_assert(MATRIX_ROW_PADDING % (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0, "Risk of out-of-bounds access.");

// --- [vecdotq.cuh:18-29, get_int_b2 dropped]  get_int_b4 (load_tiles_q4_K reads
//     weight ints via get_int_b4; the get_int_b2 sibling was only used by the
//     removed dp4a dot) ---
static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32]; // assume at least 4 byte alignment
}

// --- [verbatim vecdotq.cuh:502]  VDR for the q4_K MMQ dot ---
#define VDR_Q4_K_Q8_1_MMQ  8

// [vecdotq.cuh:530-555, REMOVED]  vec_dot_q4_K_q8_1_impl_mmq was the dp4a q4_K dot's
// inner product; with the dp4a dot gone (tensor-core-only) it is unreferenced.


// ============================================================================
// [2] MMA PRIMITIVES  (NVIDIA Turing/Ampere(/Blackwell), derived from ggml/src/ggml-cuda/mma.cuh)
// ============================================================================
// Originally vendored from mma.cuh (lines 21-1410; the leading #pragma once +
// #include "common.cuh" were dropped — the shim above supplies common.cuh's
// symbols). NO LONGER a verbatim copy: every architecture #if/#elif/#else ladder
// has been collapsed to the live NVIDIA Turing/Ampere arm, deleting the
// AMD MFMA/WMMA (RDNA*/CDNA/GCN/GFX) and the LEGACY Volta (sm_70) arms
// ("cut off the past, keep the future" — the forward-looking >= AMPERE / Blackwell
// branches are kept). The int8 m16n8k32 mma (tile<16,8,int> A, tile<8,8,int> B)
// is what the q4_K MMA dot uses on sm_86; only the DATA_LAYOUT_I_MAJOR int tiles +
// their load_ldmatrix/load_generic/movmatrix are on the live path. The dead
// half2/bf16 tiles, the J_MAJOR / *_MIRRORED (CDNA/RDNA/Volta) layout tiles, and
// the f16/bf16/tf32/f32/Volta/AMD mma() overloads are pruned at whole-struct /
// whole-overload granularity. Behaviorally BIT-IDENTICAL to the pristine vendor
// copy on the live sm_86 path (verified via memcmp == 0 across the stream-K /
// need_check / fixup shapes).

#if CUDART_VERSION >= 11080

static __device__ __forceinline__ int ggml_cuda_movmatrix(const int x) {
    int ret = 0;

#ifdef TURING_MMA_AVAILABLE
    asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;"
        : "=r"(ret) : "r"(x));
#endif // defined(TURING_MMA_AVAILABLE)
    return ret;
}

#else

static __device__ __forceinline__ int ggml_cuda_movmatrix(const int x) {
    // Imagine transposing row-major matrix to column-major matrix.
    const int src_i_low  = 2 * (threadIdx.x % 4);
    const int src_i_high = src_i_low + 1;
    const int src_j      = threadIdx.x / 4;

    const int src_laneid_low  = src_i_low  * 4 + src_j / 2;
    const int src_laneid_high = src_i_high * 4 + src_j / 2;

    const int shift_low  = ((src_j + 0) % 2) * 16;
    const int shift_high = ((src_j + 1) % 2) * 16;

    const int ret_low  = (__shfl_sync(0xFFFFFFFF, x, src_laneid_low,  WARP_SIZE) >> shift_low)  & 0x0000FFFF;
    const int ret_high = (__shfl_sync(0xFFFFFFFF, x, src_laneid_high, WARP_SIZE) << shift_high) & 0xFFFF0000;

    return ret_low | ret_high;
}

#endif // CUDART_VERSION >= 11080

static __device__ __forceinline__ half2 ggml_cuda_movmatrix(const half2 x) {
    half2 ret;
    *((int *) &ret) = ggml_cuda_movmatrix(*((const int *) &x));
    return ret;
}

namespace ggml_cuda_mma {

    // Some architectures like Volta or CDNA3 perform multiple matrix multiplications per warp in parallel,
    //     effectively the warp is being split into subgroups of threads that each perform a single mma instruction.
    // In those cases the data can be split in different ways across the warp.
    enum data_layout {
        // By default the data uses the I direction as its major dimension and the J direction as its minor dimension.
        // For the A/C matrices this means I major == row major, J major == column major.
        // For the B matrix this means I major == column major, J major == row major.
        // MIRRORED == Each data value is held exactly once per thread subgroup.
        DATA_LAYOUT_I_MAJOR           =  0, // Always used for Turing, Ampere, Ada Lovelace, consumer Blackwell, matrix A&B for RDNA4 and CDNA.
        DATA_LAYOUT_J_MAJOR           = 10, // Matrix C for CDNA and RDNA4, int and float matrix C for RDNA3.
        DATA_LAYOUT_I_MAJOR_MIRRORED  = 20, // Volta, matrix A&B for RDNA3.
        DATA_LAYOUT_J_MAJOR_MIRRORED  = 30,
    };
    // Implemented mma combinations are:
    //   - (I_MAJOR, I_MAJOR)          -> I_MAJOR
    //   - (I_MAJOR, I_MAJOR_MIRRORED) -> I_MAJOR
    //   - (I_MAJOR, J_MAJOR_MIRRORED) -> I_MAJOR

    static constexpr bool is_i_major(const data_layout dl) {
        return dl == DATA_LAYOUT_I_MAJOR ||
               dl == DATA_LAYOUT_I_MAJOR_MIRRORED;
    }

    static constexpr __device__ data_layout get_input_data_layout() {
        return DATA_LAYOUT_I_MAJOR;
    }

    template <int I_, int J_, typename T, data_layout ds_=DATA_LAYOUT_I_MAJOR>
    struct tile {};

    template <int I_, int J_, typename T>
    struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR;

        static constexpr int ne = I * J / 32;
        T x[ne] = {0};

        static constexpr __device__ bool supported() {
            if (I ==  8 && J ==  4) return true;
            if (I ==  8 && J ==  8) return true;
            if (I == 16 && J ==  8) return true;
            if (I == 16 && J == 16) return true;
            if (I == 32 && J ==  8) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 8 && J == 4) {
                return threadIdx.x / 4;
            } else if constexpr (I == 8 && J == 8) {
                return threadIdx.x / 4;
            } else if constexpr (I == 16 && J == 8) {
                return ((l / 2) * 8) + (threadIdx.x / 4);
            } else if constexpr (I == 16 && J == 16) {
                return (((l / 2) % 2) * 8) + (threadIdx.x / 4);
            } else if constexpr (I == 32 && J == 8) {
                return tile<16, 8, T>::get_i(l); // Memory layout simply repeated with same pattern in i direction.
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 8 && J == 4) {
                return threadIdx.x % 4;
            } else if constexpr (I == 8 && J == 8) {
                return (l * 4) + (threadIdx.x % 4);
            } else if constexpr (I == 16 && J == 8) {
                return ((threadIdx.x % 4) * 2) + (l % 2);
            } else if constexpr (I == 16 && J == 16) {
                return ((l / 4) * 8) + ((threadIdx.x % 4) * 2) + (l % 2);
            } else if constexpr (I == 32 && J == 8) {
                return tile<16, 8, T>::get_j(l); // Memory layout simply repeated with same pattern in i direction.
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
    };

    // [PRUNED] the tile<half2, DATA_LAYOUT_I_MAJOR> and
    //   tile<nv_bfloat162, DATA_LAYOUT_I_MAJOR> specializations (mma.cuh): the f16
    //   and bf16 MMA tiles. The int8 q4_K MMA dot uses only the int tiles
    //   (tile<16,8,int> A, tile<8,8,int> B); the float-pair tiles are unreferenced
    //   on this path. Removed as whole specializations; compiles + bit-identical
    //   without them.

    // [PRUNED] the DATA_LAYOUT_J_MAJOR / DATA_LAYOUT_I_MAJOR_MIRRORED /
    //   DATA_LAYOUT_J_MAJOR_MIRRORED tile<> specializations (mma.cuh): J_MAJOR is
    //   the CDNA/RDNA4 matrix-C layout, the *_MIRRORED layouts are Volta / RDNA3.
    //   The int8 q4_K MMA path uses only DATA_LAYOUT_I_MAJOR tiles (the tile<>
    //   template defaults dl=DATA_LAYOUT_I_MAJOR), so these are never instantiated
    //   on the NVIDIA Turing/Ampere/Blackwell target. Removed as whole
    //   specializations; compiles + bit-identical without them.

    // [PRUNED] get_half2 / get_transposed / make_identity_mat (mma.cuh:653-715):
    // half2/identity tile builders for the f16/RDNA4 MMA paths; unused by the int8
    // q4_K path. Removed as whole functions; compiles + bit-identical without them.

    template <int I, int J, typename T, data_layout dl>
    static __device__ __forceinline__ void load_generic(tile<I, J, T, dl> & t, const T * __restrict__ xs0, const int stride) {
#pragma unroll
        for (int l = 0; l < t.ne; ++l) {
            t.x[l] = xs0[t.get_i(l)*stride + t.get_j(l)];
        }
    }

    template <typename T>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<8, 8, T> & t, const T * __restrict__ xs0, const int stride) {
#ifdef TURING_MMA_AVAILABLE
        int * xi = (int *) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride + ((threadIdx.x / t.I) * (t.J / 2)) % t.J;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
            : "=r"(xi[0]), "=r"(xi[1])
            : "l"(xs));
#else
        load_generic(t, xs0, stride);
#endif // TURING_MMA_AVAILABLE
    }

    template <typename T>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 4, T> & t, const T * __restrict__ xs0, const int stride) {
#ifdef TURING_MMA_AVAILABLE
        int * xi = (int *) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
            : "=r"(xi[0]), "=r"(xi[1])
            : "l"(xs));
#else
        load_generic(t, xs0, stride);
#endif // TURING_MMA_AVAILABLE
    }

    template <typename T, data_layout dl>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 8, T, dl> & t, const T * __restrict__ xs0, const int stride) {
#if defined(TURING_MMA_AVAILABLE)
        int * xi = (int * ) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J / 2);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
            : "l"(xs));
#else
        load_generic(t, xs0, stride);
#endif // TURING_MMA_AVAILABLE
    }

    // [PRUNED] half2-tile load_ldmatrix overloads (mma.cuh:815-836) for
    //   tile<8,4,half2,*_MIRRORED> and tile<32,4,half2> (Volta/RDNA3 f16 paths),
    //   and load_ldmatrix_trans (mma.cuh:838-851). Unused by the int8 q4_K path.

    // [PRUNED] mma(tile<16,8,int>, tile<16,4,int>, tile<8,4,int>) — the int8
    //   m16n8k16 overload (mma.cuh:853-873). The q4_K MMA dot uses the m16n8k32
    //   overload below (tile<16,8,int> A, tile<8,8,int> B); this k16 sibling is
    //   unreferenced. Removed; compiles + bit-identical without it.

    static __device__ __forceinline__ void mma(
            tile<16, 8, int> & D, const tile<16, 8, int> & A, const tile<8, 8, int> & B) {
        // Ampere floor: m16n8k32.s8 is sm_80+. The Turing 4x m8n8k16 fallback was
        // dropped — little-gemma builds for Ampere and forward (fa_mma's m16n8k16.f16
        // and cp.async already require sm_80, so a sub-Ampere build never compiles).
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
            : "r"(A.x[0]), "r"(A.x[1]), "r"(A.x[2]), "r"(A.x[3]), "r"(B.x[0]), "r"(B.x[1]));
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    }

    // [PRUNED] all remaining mma() overloads (mma.cuh:875-1409, except the int8
    //   m16n8k32 kept above): the f16/bf16/tf32/f32 tensor-core overloads, the
    //   AMD MFMA/WMMA int and half overloads, the Volta m8n8k4 overloads, the
    //   block-scaled FP4 overload, and the 32-row reinterpret wrappers. None are
    //   reachable from the int8 q4_K MMA dot. Removed as whole overloads;
    //   compiles + bit-identical without them.

}

// ============================================================================
// [3] q4_K MMQ KERNEL  (VERBATIM from ggml/src/ggml-cuda/mmq.cuh)
// ============================================================================
// Pruned to the q4_K MMA path. Kept ranges are cited per block. All other
// load_tiles_*/vec_dot_* (q1_0/q4_0/q4_1/q5_0/q5_1/q8_0/mxfp4/nvfp4/q8_0_16/
// q2_K/q3_K/q5_K/q6_K/iq*), every non-Q4_K mmq_type_traits specialization,
// mmq_args, launch_matmul_q_prefill, matmul_q_prefill_case, and the ggml_cuda_op_* host
// launchers are PRUNED (lg_mmq_q4k in section [5] replaces the launchers).

// --- [verbatim mmq.cuh:10] ---
using namespace ggml_cuda_mma;

// --- [verbatim mmq.cuh:12-307]  constants, typedefs, block_q8_1_mmq +
//     block_fp4_mmq + static_asserts, mmq_get_q8_1_ds_layout, tile_x_sizes,
//     get_mmq_x_max host/device, get_mmq_y host/device, get_iter_k, the
//     MMQ_DP4A_TXS_* / MMQ_MMA_TILE_X_K_* macros + asserts,
//     mmq_get_dp4a_tile_x_sizes, mmq_get_mma_tile_x_k, MMQ_TILE_Y_K,
//     granularity host/device, nwarps host/device.
// ----------------------------------------------------------------------------
#define MMQ_DP4A_MAX_BATCH_SIZE 64 // Max. batch size to use for dp4a MMQ kernels when FP16 tensor cores are available.
#define MMQ_ITER_K 256
#define MMQ_ITER_K_MXFP4_FP4    512
#define MMQ_NWARPS 8

typedef void (*load_tiles_mmq_t)(const char * __restrict__ x, int * x_tile, const int kbx0, const int i_max, const int stride);
typedef void (*vec_dot_mmq_t)(const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00);
typedef void (*mmq_write_back_t)(const float * __restrict__ sum, const int32_t * __restrict__ get_rows_to_sorted,
    float * __restrict__ dst, const int stride, const int i_max, const int j_max);

enum mmq_q8_1_ds_layout {
    MMQ_Q8_1_DS_LAYOUT_D4,
    MMQ_Q8_1_DS_LAYOUT_DS4,
    MMQ_Q8_1_DS_LAYOUT_D2S6,
};

struct block_q8_1_mmq {
    // The y float data is converted to a data layout that can simply be copied to shared memory as a contiguous block.
    // The y float data is first grouped as blocks of 128 values.
    // These blocks are then treated as individual data values and transposed.
    //
    // To avoid shared memory bank conflicts each block is padded with 16 bytes.
    // This padding is also used to store block scales/partial sums.
    // The scales multiplied with the quantized data are equal to the unquantized values.
    // The partial sums are obtained by summing up a subgroup of the contained values (prior to quantization)
    //     and are only needed for performance reasons.
    //
    // The exact data stored depends on the x data type.
    union {
        float d4[4];    // 1 32 bit scale per 32 values, stored as d0,d1,d2,d3
        half2 ds4[4];   // 1 16 bit scale + 1 16 bit partial sum per 32 values, stored as d0,s0,d1,s1,d2,s2,d3,s3
        half  d2s6[8];  // 1 16 bit scale per 64 values + 1 16 bit partial sum per 16 values for the first 96 values,
                        //     stored as d0,d1,s1,s2,s3,s4,s5
    };
    int8_t qs[4*QK8_1]; // 128 values quantized to 8 bit each
};

struct block_fp4_mmq {
    uint32_t d4[4];       // 8 E8M0 scales (1 per 32 values), 2 packed per uint32: d4[0]={s0,s1}, d4[1]={s2,s3}, etc.
    int8_t   qs[4 * 32];  // 256 FP4 values packed as 4-bit pairs (2 per byte), 8 blocks of 32 values
};

static_assert(sizeof(block_q8_1_mmq) == 4*QK8_1 + 4*sizeof(half2), "Unexpected block_q8_1_mmq size");
static_assert(sizeof(block_q8_1_mmq) == 4*sizeof(block_q8_1),      "Unexpected block_q8_1_mmq size");
static_assert(sizeof(block_fp4_mmq)  == sizeof(block_q8_1_mmq),    "Unexpected block_fp4_mmq size");

static mmq_q8_1_ds_layout mmq_get_q8_1_ds_layout(const ggml_type type_x) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_Q5_0:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q5_1:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_Q8_0:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_MXFP4:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_NVFP4:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q2_K:
            return MMQ_Q8_1_DS_LAYOUT_D2S6;
        case GGML_TYPE_Q3_K:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        case GGML_TYPE_IQ1_S:
            return MMQ_Q8_1_DS_LAYOUT_DS4;
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            return MMQ_Q8_1_DS_LAYOUT_D4;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

struct tile_x_sizes {
    int qs;
    int dm;
    int sc;
};

static int get_mmq_x_max_host(const int cc) {
    return turing_mma_available(cc) ? 128 :
        GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA ?
#ifdef GGML_CUDA_FORCE_MMQ
            128                     : 64;
#else
            MMQ_DP4A_MAX_BATCH_SIZE : 64;
#endif // GGML_CUDA_FORCE_MMQ
}

// [mmq.cuh, AMD/HIP arms dropped, legacy (< Volta, returned 64) removed]
// Tensor-core-only: every supported target (Ampere+) has TURING_MMA_AVAILABLE,
// so return the MMA value unconditionally (the dp4a/force-mmq arms are unreachable).
static constexpr __device__ int get_mmq_x_max_device() {
    return 128;
}

static int get_mmq_y_host(const int cc) {
    return (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) ? 128 : 64;
}

static constexpr __device__ int get_iter_k([[maybe_unused]] const ggml_type type) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    return type == GGML_TYPE_MXFP4 ? MMQ_ITER_K_MXFP4_FP4 : MMQ_ITER_K;
#else
    return MMQ_ITER_K;
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

// [mmq.cuh, HIP/RDNA1 arms dropped, legacy (< Volta, returned 64) removed]
static constexpr __device__ int get_mmq_y_device() {
    return 128;
}

// Decouple shared memory tile sizes from WARP_SIZE to allow for different warp sizes.
// The K dimension of the tiles has either,
// 1*MMQ_TILE_NE_K==32 (always for TILE_Y_K) or 2*MMQ_TILE_NE_K==64 (typically for TILE_X_K),
// 32 bit elements for the quantized data (does not include scales).
// In other words, the size of the quantized data in the K dimension is a multiple of MMQ_TILE_NE_K.
// The final tile size in K direction is padded to avoid shared memory bank conflicts,
// in terms of 32 bit elements that means K % 2 == 1 for dp4a or K % 8 == 4 for mma.
#define MMQ_TILE_NE_K 32

#define MMQ_DP4A_TXS_Q4_0    tile_x_sizes{mmq_y*MMQ_TILE_NE_K   + mmq_y, mmq_y*MMQ_TILE_NE_K/QI4_0   + mmq_y/QI4_0,     0}
#define MMQ_DP4A_TXS_Q4_1    tile_x_sizes{mmq_y*MMQ_TILE_NE_K   + mmq_y, mmq_y*MMQ_TILE_NE_K/QI4_1   + mmq_y/QI4_1,     0}
#define MMQ_DP4A_TXS_Q8_0    tile_x_sizes{mmq_y*MMQ_TILE_NE_K*2 + mmq_y, mmq_y*MMQ_TILE_NE_K*2/QI8_0 + mmq_y/(QI8_0/2), 0}
#define MMQ_DP4A_TXS_Q8_0_16 tile_x_sizes{mmq_y*MMQ_TILE_NE_K*2 + mmq_y, mmq_y*MMQ_TILE_NE_K*4/QI8_0 + mmq_y/(QI8_0/4), 0}
#define MMQ_DP4A_TXS_Q8_1    tile_x_sizes{mmq_y*MMQ_TILE_NE_K*2 + mmq_y, mmq_y*MMQ_TILE_NE_K*2/QI8_1 + mmq_y/(QI8_1/2), 0}
#define MMQ_DP4A_TXS_Q2_K    tile_x_sizes{mmq_y*MMQ_TILE_NE_K*2 + mmq_y, mmq_y*MMQ_TILE_NE_K         + mmq_y,           0}
#define MMQ_DP4A_TXS_Q3_K    tile_x_sizes{mmq_y*MMQ_TILE_NE_K*2 + mmq_y, mmq_y,                                         mmq_y*MMQ_TILE_NE_K/8 + mmq_y/8}
#define MMQ_DP4A_TXS_Q4_K    tile_x_sizes{mmq_y*MMQ_TILE_NE_K   + mmq_y, mmq_y*MMQ_TILE_NE_K/QI4_K,                     mmq_y*MMQ_TILE_NE_K/8 + mmq_y/8}
#define MMQ_DP4A_TXS_Q5_K    tile_x_sizes{mmq_y*MMQ_TILE_NE_K*2 + mmq_y, mmq_y*MMQ_TILE_NE_K/QI5_K   + mmq_y/QI5_K,     mmq_y*MMQ_TILE_NE_K/8 + mmq_y/8}
#define MMQ_DP4A_TXS_Q6_K    tile_x_sizes{mmq_y*MMQ_TILE_NE_K*2 + mmq_y, mmq_y*MMQ_TILE_NE_K/QI6_K   + mmq_y/QI6_K,     mmq_y*MMQ_TILE_NE_K/8 + mmq_y/8}

static constexpr __host__ __device__ tile_x_sizes mmq_get_dp4a_tile_x_sizes(ggml_type type, int mmq_y) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_Q4_0:    return MMQ_DP4A_TXS_Q4_0;
        case GGML_TYPE_Q4_1:    return MMQ_DP4A_TXS_Q4_1;
        case GGML_TYPE_Q5_0:    return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_Q5_1:    return MMQ_DP4A_TXS_Q8_1;
        case GGML_TYPE_Q8_0:    return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_MXFP4:   return MMQ_DP4A_TXS_Q8_1;
        case GGML_TYPE_NVFP4:   return MMQ_DP4A_TXS_Q8_0_16;
        case GGML_TYPE_Q2_K:    return MMQ_DP4A_TXS_Q2_K;
        case GGML_TYPE_Q3_K:    return MMQ_DP4A_TXS_Q3_K;
        case GGML_TYPE_Q4_K:    return MMQ_DP4A_TXS_Q4_K;
        case GGML_TYPE_Q5_K:    return MMQ_DP4A_TXS_Q5_K;
        case GGML_TYPE_Q6_K:    return MMQ_DP4A_TXS_Q6_K;
        case GGML_TYPE_IQ2_XXS: return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ2_XS:  return MMQ_DP4A_TXS_Q8_0_16;
        case GGML_TYPE_IQ2_S:   return MMQ_DP4A_TXS_Q8_0_16;
        case GGML_TYPE_IQ3_XXS: return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ3_S:   return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ1_S:   return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ4_XS:  return MMQ_DP4A_TXS_Q8_0;
        case GGML_TYPE_IQ4_NL:  return MMQ_DP4A_TXS_Q8_0;
        default:                return tile_x_sizes{0, 0, 0};
    }
}

#define MMQ_MMA_TILE_X_K_Q8_0  (2*MMQ_TILE_NE_K + 2*MMQ_TILE_NE_K/QI8_0                   + 4)
#define MMQ_MMA_TILE_X_K_FP4   (2*MMQ_TILE_NE_K + 8                                       + 4) // MXFP4
#define MMQ_MMA_TILE_X_K_NVFP4 (2*MMQ_TILE_NE_K + MMQ_TILE_NE_K/2                         + 4) // NVFP4
#define MMQ_MMA_TILE_X_K_Q8_1  (2*MMQ_TILE_NE_K + 2*MMQ_TILE_NE_K/QI8_0                   + 4)
#define MMQ_MMA_TILE_X_K_Q2_K  (2*MMQ_TILE_NE_K + MMQ_TILE_NE_K                           + 4)
#define MMQ_MMA_TILE_X_K_Q3_K  (2*MMQ_TILE_NE_K + MMQ_TILE_NE_K/2                         + 4)
#define MMQ_MMA_TILE_X_K_Q6_K  (2*MMQ_TILE_NE_K + MMQ_TILE_NE_K/QI6_K   + MMQ_TILE_NE_K/8 + 7)

static_assert(MMQ_MMA_TILE_X_K_Q8_0 % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q8_1 % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q2_K % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q3_K % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q6_K % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_FP4  % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_FP4 == MMQ_MMA_TILE_X_K_Q8_1, "Wrong tile size for MXFP4");
static_assert(MMQ_MMA_TILE_X_K_NVFP4 % 8 == 4, "Wrong padding.");


static constexpr __host__ __device__ int mmq_get_mma_tile_x_k(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_Q4_0:    return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_Q4_1:    return MMQ_MMA_TILE_X_K_Q8_1;
        case GGML_TYPE_Q5_0:    return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_Q5_1:    return MMQ_MMA_TILE_X_K_Q8_1;
        case GGML_TYPE_Q8_0:    return MMQ_MMA_TILE_X_K_Q8_0;
        // tile sizes are the same for Q8_1 and FP4 for blackwell
        case GGML_TYPE_MXFP4:   return MMQ_MMA_TILE_X_K_Q8_1;
        case GGML_TYPE_NVFP4:   return MMQ_MMA_TILE_X_K_NVFP4;
        case GGML_TYPE_Q2_K:    return MMQ_MMA_TILE_X_K_Q2_K;
        case GGML_TYPE_Q3_K:    return MMQ_MMA_TILE_X_K_Q3_K;
        case GGML_TYPE_Q4_K:    return MMQ_MMA_TILE_X_K_Q8_1;
        case GGML_TYPE_Q5_K:    return MMQ_MMA_TILE_X_K_Q8_1;
        case GGML_TYPE_Q6_K:    return MMQ_MMA_TILE_X_K_Q6_K;
        case GGML_TYPE_IQ2_XXS: return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_IQ2_XS:  return MMQ_MMA_TILE_X_K_Q3_K;
        case GGML_TYPE_IQ2_S:   return MMQ_MMA_TILE_X_K_Q3_K;
        case GGML_TYPE_IQ3_XXS: return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_IQ3_S:   return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_IQ1_S:   return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_IQ4_XS:  return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_IQ4_NL:  return MMQ_MMA_TILE_X_K_Q8_0;
        default:                return 0;
    }
}

// block_q8_1_mmq has (128 8-bit ints == 32 32-bit ints + 4 32-bit scales)
#define MMQ_TILE_Y_K     (MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1)
#define MMQ_TILE_Y_FP4_K MMQ_TILE_Y_K

// [mmq.cuh, AMD branch dropped]
static int mmq_get_granularity_host(const int mmq_x, const int cc) {
    if (turing_mma_available(cc) && mmq_x >= 48) {
        return 16;
    } else {
        return 8;
    }
}

// Tensor-core-only: TURING_MMA_AVAILABLE is always defined on supported targets,
// so return the MMA-path granularity unconditionally.
static constexpr __device__ int mmq_get_granularity_device(const int mmq_x) {
    return mmq_x >= 48 ? 16 : 8;
}

// [mmq.cuh, HIP arm dropped]
static int mmq_get_nwarps_host(const int /*cc*/, const int warp_size) {
    return 256/warp_size;
}

// [mmq.cuh, AMD arm dropped]
static constexpr __device__ int mmq_get_nwarps_device() {
    return 256/ggml_cuda_get_physical_warp_size();
}

// ------------------------------------------------------------

// --- [mmq.cuh:1270-1397, AMD MFMA/WMMA arm dropped]  vec_dot_q8_1_q8_1_mma
//     (the int8 MMA dot q4_K uses on the NVIDIA Turing+ path) ---
template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q8_1_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    typedef tile<16,  8, int> tile_A;
    typedef tile< 8,  8, int> tile_B;
    typedef tile<16,  8, int> tile_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + 2*MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_dm = (const half2 *) y;

    tile_A   A[ntx][MMQ_TILE_NE_K/QI8_1];
    float2 dmA[ntx][tile_C::ne/2][MMQ_TILE_NE_K/QI8_1];

    const int i0 = (threadIdx.y/ntx)*rows_per_warp;

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            const int k0 = k00 + k01;

            load_ldmatrix(A[n][k01/QI8_1], x_qs + (i0 + n*tile_A::I)*MMQ_MMA_TILE_X_K_Q8_1 + k0, MMQ_MMA_TILE_X_K_Q8_1);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_A::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
                const int k0 = k00 + k01;

                dmA[n][l][k01/QI8_1] = __half22float2(x_dm[i*MMQ_MMA_TILE_X_K_Q8_1 + k0/QI8_1]);
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            tile_B   B;
            float2 dsB[tile_C::ne/2];

            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K); // faster than load_ldmatrix

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                dsB[l] = __half22float2(y_dm[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n][k01/QI8_1], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA[n][l/2][k01/QI8_1].x*dsB[l%2].x*C.x[l];
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA[n][l/2][k01/QI8_1].y*dsB[l%2].y;
                }
            }
        }
    }
}

// --- [mmq.cuh:2141-2293, dp4a dot dropped]  unpack_scales_q45_K, load_tiles_q4_K
//     (the MMA layout arm only; vec_dot_q4_K_q8_1_dp4a removed — tensor-core-only). ---
static __device__ __forceinline__ int unpack_scales_q45_K(const int * scales, const int ksc) {
    // scale arrangement after the following two lines:
    //   - ksc == 0: sc0, sc1, sc2, sc3
    //   - ksc == 1: sc4, sc5, sc6, sc7
    //   - ksc == 2:  m0,  m1,  m2,  m3
    //   - ksc == 3:  m4,  m5,  m6,  m7
    return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) | // lower 4 bits
           ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);  // upper 2 bits
}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q4_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_K);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride;
        const int qs0 = get_int_b4(bxi->qs, txi);

        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + 16*(txi/8) + txi % 8 + 0] = (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_1 + 16*(txi/8) + txi % 8 + 8] = (qs0 >> 4) & 0x0F0F0F0F;
    }

    constexpr int rows_per_warp = warp_size / 2;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*rows_per_warp) {
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/2) % mmq_y;
        {
            if (need_check) {
                i = min(i, i_max);
            }

            const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride;

            const int * scales = (const int *) bxi->scales;
            const int ksc = threadIdx.x % 2;

            const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
            const int  m32 = unpack_scales_q45_K(scales, ksc + 2);

            const uint8_t * sc8 = (const uint8_t *) &sc32;
            const uint8_t *  m8 = (const uint8_t *)  &m32;

            const half2 dm = bxi->dm * make_half2(1.0f, -1.0f);

    #pragma unroll
            for (int l = 0; l < sizeof(int); ++l) {
                x_dm[i*MMQ_MMA_TILE_X_K_Q8_1 + sizeof(int)*ksc + l] = dm*make_half2(sc8[l], m8[l]);
            }
        }
    }
}

// --- [mmq.cuh:3296-3376, dp4a write-back dropped]  mmq_write_back_mma,
//     mmq_type_traits primary template declaration (tensor-core-only). ---
template<ggml_type type, int mmq_x, int mmq_y, bool need_check>
static __device__ __forceinline__ void mmq_write_back_mma(
        const float * __restrict__ sum, const int * __restrict__ ids_dst, float * __restrict__ dst,
        const int stride, const int i_max, const int j_max) {

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int nwarps = mmq_get_nwarps_device();

    // [mmq.cuh, AMD MFMA/WMMA arm dropped]
    typedef tile<16, 8, int> tile_C;
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_C::I);
#if defined(TURING_MMA_AVAILABLE)
    static_assert(nwarps*tile_C::I == mmq_y, "nwarps*tile_C::I != mmq_y");
#else
    GGML_UNUSED(nwarps);
#endif // defined(TURING_MMA_AVAILABLE)

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);

                if (j > j_max) {
                    continue;
                }

                const int i = i0 + n*tile_C::I + tile_C::get_i(l);

                if (need_check && i > i_max) {
                    continue;
                }

                dst[ids_dst[j]*stride + i] = sum[(j0/tile_C::J + n)*tile_C::ne + l];
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------------------------------------------

template <int mmq_x, int mmq_y, bool need_check, ggml_type type>
struct mmq_type_traits;

// --- [verbatim mmq.cuh:3463-3469]  mmq_type_traits<...,GGML_TYPE_Q4_K> ---
template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_Q4_K> {
    static constexpr int              vdr          = VDR_Q4_K_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q4_K<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y>;
};

// --- [verbatim mmq.cuh:3551-4048]  matmul_q_prefill_process_tile, matmul_q_prefill (kernel,
//     __launch_bounds__ arch #if verbatim), matmul_q_prefill_stream_k_fixup. ---
template <ggml_type type, int mmq_x, bool need_check, bool fixup>
static __device__ __forceinline__ void matmul_q_prefill_process_tile(
        const char * __restrict__ x, const int offset_x, const int * __restrict__ y,
        const int * __restrict__ ids_dst, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop) {

    constexpr int              warp_size  = ggml_cuda_get_physical_warp_size();
    constexpr int              nwarps     = mmq_get_nwarps_device();
    constexpr int              qk         = ggml_cuda_type_traits<type>::qk;
    constexpr int              mmq_y      = get_mmq_y_device();
    constexpr load_tiles_mmq_t load_tiles = mmq_type_traits<mmq_x, mmq_y, need_check, type>::load_tiles;

    extern __shared__ int data_matmul_q_prefill[];
    int * tile_y = data_matmul_q_prefill + mmq_x;
    int * tile_x = tile_y + GGML_PAD(mmq_x*MMQ_TILE_Y_K, nwarps*warp_size);

    constexpr vec_dot_mmq_t    vec_dot    = mmq_type_traits<mmq_x, mmq_y, need_check, type>::vec_dot_mma;
    constexpr mmq_write_back_t write_back = mmq_write_back_mma<type, mmq_x, mmq_y, need_check>;

#if defined(BLACKWELL_MMA_AVAILABLE)
    // FP4 tile stores 8 blocks
    constexpr int ne_block = (type == GGML_TYPE_MXFP4) ? 8 * QK_MXFP4 : 4 * QK8_1;
#else
    constexpr int ne_block = 4 * QK8_1;
#endif  // defined(BLACKWELL_MMA_AVAILABLE)

    constexpr int ITER_K          = get_iter_k(type);
    constexpr int blocks_per_iter = ITER_K / qk;

    float sum[mmq_x*mmq_y / (nwarps*warp_size)] = {0.0f};

    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        {
            const int * by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;
#pragma unroll
            for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y*warp_size + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        vec_dot(tile_x, tile_y, sum, 0);

        __syncthreads();

        {
            const int * by0 = y + ncols_y * ((kb0 * qk / ne_block) * sz + sz);
#pragma unroll
            for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y*warp_size + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);

        __syncthreads();
    }

    if (fixup) {
        write_back(sum, ids_dst, tmp_fixup + blockIdx.x*(mmq_x*mmq_y), mmq_y, mmq_y, mmq_x);
    } else {
        write_back(sum, ids_dst, dst, stride_col_dst, tile_x_max_i, tile_y_max_j);
    }
}


// The matmul_q_prefill kernel implements "stream-k" work partitioning as described in https://arxiv.org/abs/2301.03598

template <ggml_type type, int mmq_x, bool need_check>
// [mmq.cuh, HIP arm dropped, legacy (< Volta, min-blocks 2) removed]
    __launch_bounds__(ggml_cuda_get_physical_warp_size()*mmq_get_nwarps_device(), 1)
static __global__ void matmul_q_prefill(
        const char * __restrict__ x, const int * __restrict__ y, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const int ncols_x, const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int channel_ratio, const int nchannels_y, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int sample_ratio, const int nsamples_y, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ncols_max) {

    // Skip unused template specializations for faster compilation:
    if (mmq_x > get_mmq_x_max_device() || mmq_x % mmq_get_granularity_device(mmq_x) != 0) {
        NO_DEVICE_CODE;
        return;
    }

    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr int qk    = ggml_cuda_type_traits<type>::qk;
    constexpr int mmq_y = get_mmq_y_device();

    const int ntx = (ncols_max + mmq_x - 1) / mmq_x; // Number of tiles x
    const int nty = (nrows_x   + mmq_y - 1) / mmq_y; // Number of tiles y

    // Initialize the ids for writing back data with just the index.
    // For regular matrix multiplications this is never changed.
    // For MoE the correct indices are loaded from ids_dst.
    extern __shared__ int ids_dst_shared[]; // Stored at beginning of shared memory.
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps*warp_size) {
        const int j = j0 + threadIdx.y*warp_size + threadIdx.x;

        if (j0 + nwarps*warp_size > mmq_x && j >= mmq_x) {
            break;
        }

        ids_dst_shared[j] = j;
    }
    __syncthreads();

    // [mmq.cuh] The HIP-non-CDNA / legacy-(< Volta) conventional-tiling fast path
    // was removed: on the supported NVIDIA target stream-k is always used.

    constexpr int ITER_K = get_iter_k(type);

    const     int64_t blocks_per_ne00 = ncols_x / qk;
    constexpr int     blocks_per_iter = ITER_K / qk;

    // kbc == k block continuous, current index in continuous ijk space.
    int64_t kbc      = (int64_t) blockIdx.x     *nsamples_y*nchannels_y*ntx*nty*blocks_per_ne00 / gridDim.x;
    int64_t kbc_stop = (int64_t)(blockIdx.x + 1)*nsamples_y*nchannels_y*ntx*nty*blocks_per_ne00 / gridDim.x;

    kbc      -= (kbc      % blocks_per_ne00) % blocks_per_iter;
    kbc_stop -= (kbc_stop % blocks_per_ne00) % blocks_per_iter;

    // kb0 == k index when doing the matrix multiplication for an output tile.
    int kb0_start = kbc % blocks_per_ne00;
    int kb0_stop  = min(blocks_per_ne00, kb0_start + kbc_stop - kbc);
    while (kbc < kbc_stop && kb0_stop == blocks_per_ne00) {
        int tmp = kbc;
        const int it = tmp / (nsamples_y*nchannels_y*ntx*blocks_per_ne00);
        tmp -= it * (nsamples_y*nchannels_y*ntx*blocks_per_ne00);
        const int wt = tmp / (nchannels_y*ntx*blocks_per_ne00);
        tmp -= wt * (nchannels_y*ntx*blocks_per_ne00);
        const int zt = tmp / (ntx*blocks_per_ne00);
        tmp -= zt * (ntx*blocks_per_ne00);
        const int jt = tmp / blocks_per_ne00;

        // Defaults for regular matrix multiplication:
        int col_low    = 0;
        int col_high   = ncols_dst;
        int col_diff   = ncols_dst;
        int offset_y   = wt*stride_sample_y   + zt*stride_channel_y;
        int offset_dst = wt*stride_sample_dst + zt*stride_channel_dst + jt*mmq_x*stride_col_dst;

        if (ids_dst) {
            col_low  = expert_bounds[zt + 0];
            col_high = expert_bounds[zt + 1];
            col_diff = col_high - col_low;

            offset_y   = 0;
            offset_dst = 0;

            if (jt*mmq_x >= col_diff) {
                kbc += blocks_per_ne00;
                kbc -= kbc % blocks_per_ne00;

                kb0_start = 0;
                kb0_stop  = min(blocks_per_ne00, kbc_stop - kbc);

                continue;
            }

            __syncthreads();
#pragma unroll
            for (int j0 = 0; j0 < mmq_x; j0 += nwarps*warp_size) {
                const int j = j0 + threadIdx.y*warp_size + threadIdx.x;

                if (j0 + nwarps*warp_size > mmq_x && j >= mmq_x) {
                    break;
                }

                ids_dst_shared[j] = ids_dst[col_low + jt*mmq_x + j];
            }
            __syncthreads();
        }

        offset_y += (col_low + jt * mmq_x) * (sizeof(block_q8_1_mmq) / sizeof(int));
        offset_dst += it*mmq_y;

        const int tile_x_max_i = nrows_x  - it*mmq_y - 1;
        const int tile_y_max_j = col_diff - jt*mmq_x - 1;

        const int offset_x = (wt/sample_ratio)*stride_sample_x + (zt/channel_ratio)*stride_channel_x + it*mmq_y*stride_row_x;

        constexpr bool fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        matmul_q_prefill_process_tile<type, mmq_x, need_check, fixup>
            (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, stride_row_x, ncols_y, stride_col_dst,
             tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);

        kbc += blocks_per_ne00;
        kbc -= kbc % blocks_per_ne00;

        kb0_start = 0;
        kb0_stop  = min(blocks_per_ne00, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    int tmp = kbc;
    const int it = tmp / (nsamples_y*nchannels_y*ntx*blocks_per_ne00);
    tmp -= it * (nsamples_y*nchannels_y*ntx*blocks_per_ne00);
    const int wt = tmp / (nchannels_y*ntx*blocks_per_ne00);
    tmp -= wt * (nchannels_y*ntx*blocks_per_ne00);
    const int zt = tmp / (ntx*blocks_per_ne00);
    tmp -= zt * (ntx*blocks_per_ne00);
    const int jt = tmp / blocks_per_ne00;

    // Defaults for regular matrix multiplication:
    int col_low    = 0;
    int col_high   = ncols_dst;
    int col_diff   = ncols_dst;
    int offset_y   = wt*stride_sample_y   + zt*stride_channel_y;
    int offset_dst = wt*stride_sample_dst + zt*stride_channel_dst + jt*mmq_x*stride_col_dst;

    if (ids_dst) {
        col_low  = expert_bounds[zt + 0];
        col_high = expert_bounds[zt + 1];
        col_diff = col_high - col_low;

        offset_y   = 0;
        offset_dst = 0;

        if (jt*mmq_x >= col_diff) {
            return;
        }

        // The memory layout for the fixup buffer is always contiguous, therefore reset ids:
        __syncthreads();
#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps*warp_size) {
            const int j = j0 + threadIdx.y*warp_size + threadIdx.x;

            if (j0 + nwarps*warp_size > mmq_x && j >= mmq_x) {
                break;
            }

            ids_dst_shared[j] = j;
        }
        __syncthreads();
    }

    offset_y += (col_low + jt * mmq_x) * (sizeof(block_q8_1_mmq) / sizeof(int));
    offset_dst += it*mmq_y;

    const int tile_x_max_i = nrows_x  - it*mmq_y - 1;
    const int tile_y_max_j = col_diff - jt*mmq_x - 1;

    const int offset_x = (wt/sample_ratio)*stride_sample_x + (zt/channel_ratio)*stride_channel_x + it*mmq_y*stride_row_x;

    constexpr bool fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    matmul_q_prefill_process_tile<type, mmq_x, need_check, fixup>
        (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, stride_row_x, ncols_y, stride_col_dst,
         tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);
}

template <ggml_type type, int mmq_x, bool need_check>
static __global__ void matmul_q_prefill_stream_k_fixup(const int32_t * ids_dst,
                                                const int32_t * expert_bounds,
                                                float * __restrict__ dst,
                                                const float * __restrict__ tmp_last_tile,
                                                const int    ncols_x,
                                                const int    nrows_x,
                                                const int    ncols_dst,
                                                const size_t stride_col_dst,
                                                const int    nchannels_y,
                                                const size_t stride_channel_dst,
                                                const int    nsamples_y,
                                                const size_t stride_sample_dst,
                                                const int    ncols_max) {
    constexpr int     mmq_y           = get_mmq_y_device();
    constexpr int     qk              = ggml_cuda_type_traits<type>::qk;
    constexpr int     ITER_K          = get_iter_k(type);

    constexpr int     blocks_per_iter = ITER_K / qk;
    const     int64_t blocks_per_ne00 = ncols_x / qk;

    constexpr int nwarps = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    float sum[mmq_x*mmq_y / (nwarps*warp_size)] = {0.0f};

    const int ntx  = (ncols_max + mmq_x - 1) / mmq_x;
    const int nty  = (nrows_x   + mmq_y - 1) / mmq_y;

    const int bidx0 = blockIdx.x;

    // kbc == k block continuous, current index in continuous ijk space.
    int64_t kbc0      = (int64_t) bidx0     *nsamples_y*nchannels_y*ntx*nty*blocks_per_ne00 / gridDim.x;
    int64_t kbc0_stop = (int64_t)(bidx0 + 1)*nsamples_y*nchannels_y*ntx*nty*blocks_per_ne00 / gridDim.x;

    kbc0      -= (kbc0      % blocks_per_ne00) % blocks_per_iter;
    kbc0_stop -= (kbc0_stop % blocks_per_ne00) % blocks_per_iter;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = kbc0 % blocks_per_ne00 == 0;
    const bool did_not_write_last      = kbc0/blocks_per_ne00 == kbc0_stop/blocks_per_ne00 && kbc0_stop % blocks_per_ne00 != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    bool any_fixup = false;

    // Iterate over previous blocks and sum up partial sums written to fixup buffer.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    int64_t bidx = bidx0 - 1;
    int64_t kbc_stop = kbc0;
    while(true) {
        int64_t kbc = bidx*nsamples_y*nchannels_y*ntx*nty*blocks_per_ne00 / gridDim.x;
        kbc -= (kbc % blocks_per_ne00) % blocks_per_iter;

        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        any_fixup = true;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                sum[(j0/nwarps) * (mmq_y/warp_size) + i0/warp_size] += tmp_last_tile[bidx*(mmq_x*mmq_y) + j*mmq_y + i];
            }
        }

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (kbc % blocks_per_ne00 == 0 || kbc/blocks_per_ne00 < kbc0/blocks_per_ne00) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    if (!any_fixup) {
        return;
    }

    int tmp = kbc0;
    const int it = tmp / (nsamples_y*nchannels_y*ntx*blocks_per_ne00);
    tmp -= it * (nsamples_y*nchannels_y*ntx*blocks_per_ne00);
    const int wt = tmp / (nchannels_y*ntx*blocks_per_ne00);
    tmp -= wt * (nchannels_y*ntx*blocks_per_ne00);
    const int zt = tmp / (ntx*blocks_per_ne00);
    tmp -= zt * (ntx*blocks_per_ne00);
    const int jt = tmp / blocks_per_ne00;

    if (!ids_dst) {
        const int offset_dst = wt*stride_sample_dst + zt*stride_channel_dst + jt*mmq_x*stride_col_dst + it*mmq_y;
        dst += offset_dst;

        const int i_max = nrows_x   - it*mmq_y - 1;
        const int j_max = ncols_dst - jt*mmq_x - 1;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j > j_max) {
                return;
            }

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                if (need_check && i > i_max) {
                    continue;
                }

                dst[j*stride_col_dst + i] += sum[(j0/nwarps) * (mmq_y/warp_size) + i0/warp_size];
            }
        }
        return;
    }

    __shared__ int ids_dst_shared[mmq_x];
    const int col_low  = expert_bounds[zt + 0];
    const int col_high = expert_bounds[zt + 1];
    const int col_diff = col_high - col_low;

    for (int j = threadIdx.y*warp_size + threadIdx.x; j < mmq_x; j += nwarps*warp_size) {
        ids_dst_shared[j] = ids_dst[col_low + jt*mmq_x + j];
    }
    __syncthreads();

    const int offset_dst = it*mmq_y;
    dst += offset_dst;

    const int i_max = nrows_x  - it*mmq_y - 1;
    const int j_max = col_diff - jt*mmq_x - 1;

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        if (j > j_max) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
            const int i = i0 + threadIdx.x;

            if (need_check && i > i_max) {
                continue;
            }

            dst[ids_dst_shared[j]*stride_col_dst + i] += sum[(j0/nwarps) * (mmq_y/warp_size) + i0/warp_size];
        }
    }
}

// --- [mmq.cuh:4058-4066, AMD predicates dropped]  mmq_get_nbytes_shared ---
template<ggml_type type>
static size_t mmq_get_nbytes_shared(const int mmq_x, const int mmq_y, const int cc, const int warp_size, const int nwarps) {
    const tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, mmq_y);
    const int mmq_tile_x_k = mmq_get_mma_tile_x_k(type);
    const size_t nbs_ids = mmq_x*sizeof(int);
    const size_t nbs_x = turing_mma_available(cc) ? mmq_y*mmq_tile_x_k*sizeof(int) : txs.qs*sizeof(int) + txs.dm*sizeof(half2) + txs.sc*sizeof(int);
    const size_t nbs_y = mmq_x * (sizeof(block_q8_1_mmq));
    return nbs_ids + nbs_x + GGML_PAD(nbs_y, nwarps*warp_size*sizeof(int));
}

// ============================================================================
// [4] q8_1 MMQ QUANTIZER  (VERBATIM from ggml/src/ggml-cuda/quantize.cu)
// ============================================================================
// The q8_1-MMQ activation quantizer lg_mmq_q4k calls. Pruned: quantize_q8_1 +
// quantize_row_q8_1_cuda (the row quantizer; would have pulled in fastdiv /
// warp_reduce), compute_e8m0_scale, quantize_mmq_mxfp4 + quantize_mmq_mxfp4_cuda.

// --- [verbatim quantize.cu:175-271]  quantize_mmq_q8_1<ds_layout> kernel ---
template <mmq_q8_1_ds_layout ds_layout>
static __global__ void quantize_mmq_q8_1(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t i00 = i0;
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    const float4 * x4 = (const float4 *) x;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1); // first block of channel
    const int64_t ib  = ib0 + (i0 / (4*QK8_1))*ne1 + blockIdx.x;                    // block index in channel
    const int64_t iqs = i0 % (4*QK8_1);                                             // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    const float4 xi = i0 < ne00 ? x4[(i03*s03 + i02*s02 + i01*s01 + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

        // Calculate sums across vals_per_sum/4 threads.
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);

    // Write back 4 int8 values as a single 32 bit value for better memory bandwidth:
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 != 0 || iqs >= 96) {
            return;
        }

        y[ib].d2s6[2 + iqs/16] = sum;

        if (iqs % 64 != 0) {
            return;
        }

        const float d = 1.0f / d_inv;

        y[ib].d2s6[iqs/64] = d;

        return;
    }

    if (iqs % 32 != 0) {
        return;
    }

    const float d = 1.0f / d_inv;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        y[ib].ds4[iqs/32] = make_half2(d, sum);
    } else {
        y[ib].d4[iqs/32]  = d;
    }
}

// --- [verbatim quantize.cu:289-317]  quantize_mmq_q8_1_cuda host launcher ---
void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % (4*QK8_1) == 0);

    // ne1 tends to assume the highest values, therefore use it as the "x" dimension of the CUDA grid:
    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// ============================================================================
// [5] ggml STUBS + lg_mmq_q4k BRIDGE  (carried from src/flash-llama.cu)
// ============================================================================
// The one ggml host symbol still referenced on the q4_K path: ggml_abort, which
// GGML_ABORT (used in the switch defaults of mmq_get_q8_1_ds_layout and
// quantize_mmq_q8_1_cuda) expands to. Copied from flash-llama.cu:13; keeps the
// ggml.h:355-356 signature (printf-format) so GGML_ABORT compiles. The unused
// ggml_log_internal / ggml_cuda_error / turbo_innerq_publish stubs were removed.
extern "C" void ggml_abort(const char*, int, const char*, ...) { abort(); }

// [shim] flash-llama.cu uses common.cuh's CUDA_CHECK; here we inline the same
// abort-on-error check (common.cuh:158-166 routes to an abort()).
#define LG_CUDA_CHECK(err) do { cudaError_t e_ = (err); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); abort(); } } while (0)

// Route a q4_K prefill matmul through llama's matmul_q_prefill. O[n x m] = A[n x k] . W[m x k,q4_K].
// Our activation d_x is token-major [n x k] (== their A); output d_out [n x m] (== their dst[n*m+]).
// w = block_q4_K in place (our GGUF layout). Quantizes d_x->q8_1, replicates the stream-K launch.
// (carried VERBATIM from flash-llama.cu:21-48, except CUDA_CHECK -> LG_CUDA_CHECK.)
static char  *g_q8 = NULL;   static size_t g_q8cap = 0;       // q8_1 activation scratch
static float *g_mfix = NULL; static size_t g_mfixcap = 0;     // stream-K fixup scratch
void lg_mmq_q4k(float *d_out, const void *w, int k, int m, const float *d_x, int n, cudaStream_t stream) {
    constexpr ggml_type T = GGML_TYPE_Q4_K;
    const int cc = 870, nsm = 8, ws = 32, mmq_x = 128;
    int64_t kp = GGML_PAD((int64_t)k, MATRIX_ROW_PADDING);
    size_t ybytes = (size_t)n * kp * sizeof(block_q8_1) / QK8_1 + get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
    if (ybytes > g_q8cap) { cudaFree(g_q8); LG_CUDA_CHECK(cudaMalloc(&g_q8, ybytes)); g_q8cap = ybytes; }
    quantize_mmq_q8_1_cuda(d_x, nullptr, g_q8, T, k, k, (int64_t)k * n, (int64_t)k * n, kp, n, 1, 1, stream);

    const int nwarps = mmq_get_nwarps_host(cc, ws), mmq_y = get_mmq_y_host(cc);
    const int nbs = mmq_get_nbytes_shared<T>(mmq_x, mmq_y, cc, ws, nwarps);
    int64_t s01 = k / QK_K, s12 = (int64_t)n * kp * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    int ntx = (n + mmq_x - 1) / mmq_x, nty = (m + mmq_y - 1) / mmq_y;
    bool fixup = (ntx * nty) % nsm != 0;
    if (fixup) { size_t fb = (size_t)nsm * mmq_x * mmq_y * sizeof(float); if (fb > g_mfixcap) { cudaFree(g_mfix); LG_CUDA_CHECK(cudaMalloc(&g_mfix, fb)); g_mfixcap = fb; } }
    static bool raised = false;
    if (!raised) { cudaFuncSetAttribute((matmul_q_prefill<T, mmq_x, false>), cudaFuncAttributeMaxDynamicSharedMemorySize, nbs);
                   cudaFuncSetAttribute((matmul_q_prefill<T, mmq_x, true>),  cudaFuncAttributeMaxDynamicSharedMemorySize, nbs); raised = true; }
    dim3 blk(ws, nwarps, 1), grid(nsm, 1, 1);
    bool nc = (m % mmq_y != 0); float *fx = fixup ? g_mfix : nullptr;
    if (!nc) matmul_q_prefill<T, mmq_x, false><<<grid, blk, nbs, stream>>>((const char*)w, (const int*)g_q8, nullptr, nullptr, d_out, fx, k, m, n, s01, n, m, 1, 1, 0, s12, 0, 1, 1, 0, 0, 0, n);
    else     matmul_q_prefill<T, mmq_x, true ><<<grid, blk, nbs, stream>>>((const char*)w, (const int*)g_q8, nullptr, nullptr, d_out, fx, k, m, n, s01, n, m, 1, 1, 0, s12, 0, 1, 1, 0, 0, 0, n);
    if (fixup) {
        if (!nc) matmul_q_prefill_stream_k_fixup<T, mmq_x, false><<<grid, blk, 0, stream>>>(nullptr, nullptr, d_out, fx, k, m, n, m, 1, 0, 1, 0, n);
        else     matmul_q_prefill_stream_k_fixup<T, mmq_x, true ><<<grid, blk, 0, stream>>>(nullptr, nullptr, d_out, fx, k, m, n, m, 1, 0, 1, 0, n);
    }
}
