cmake_minimum_required(VERSION 3.18)

project(little-gemma C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# MTP speculation block depth (verify width), the same constant the code reads via
# include/model.h. DEFAULT 3 = a chained 2nd draft + triple verify (1.1-1.3x over 2 on
# 12B, A5000 and Orin); -DLG_MTP_N=2 builds the conservative one-draft pair verify.
# Compile-time so the verify kernel/loop stay one tight parameterized path.
set(LG_MTP_N 3 CACHE STRING "MTP speculation block depth (3 = chained default; 2 = pair verify)")
add_compile_definitions(LG_MTP_N=${LG_MTP_N})

# Warning flags for C only (so they don't reach nvcc). The genex must be
# QUOTED: unquoted, CMake splits it at the ';' and gcc gets a literal
# "-Wextra>" (found the hard way on the Orin).
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:C>:/W3>)
    # MSVC targets bare SSE2 unless told otherwise; AVX2 lets it vectorize the
    # multi-accumulator dot products in media.c (GCC at -O3 needs no nudge —
    # NEON is baseline on aarch64, so the Orin already gets this).
    add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX2>)
else()
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra>")
endif()

# media.c carries the host vision transformer (~1 TFLOP per image); fast-math
# lets the compiler turn its dot products into real SIMD reductions, which
# /fp:precise (and GCC's default) forbid because that reorders the sums. The
# sums carry no exactness contract — the oracle comparison tolerance is 1e-4.
# Scoped to this one file: the LLM side keeps strict fp.
if(MSVC)
    set_source_files_properties(src/media.c PROPERTIES COMPILE_OPTIONS "/fp:fast")
else()
    set_source_files_properties(src/media.c PROPERTIES COMPILE_OPTIONS "-ffast-math")
endif()

# Shared, backend-agnostic sources. model.c = common host setup; media.c = the
# multimodal embedders (host f32 compute, all backends share it); the compute
# backend is model-cpu.c (CPU) or model-cuda-{f32,i8}.cu (CUDA), selected per target.
set(CORE src/run.c src/gguf.c src/quant.c src/model.c src/tokenizer.c src/media.c src/mtp.c)

# --- CPU build -------------------------------------------------------------
add_executable(run ${CORE} src/cpu/model-cpu.c src/cpu/media-gpu-stub.c)   # CPU backend + no-op GPU seam
target_include_directories(run PRIVATE include src)   # src/ for mtp-internal.h, media-internal.h
if(WIN32)
    target_link_libraries(run PRIVATE ws2_32)   # AF_UNIX sockets (-s/-c)
elseif(UNIX)
    target_link_libraries(run PRIVATE m)        # expf/sqrtf/... live in libm
endif()

# OpenMP carries media.c's host vision transformer (~1 TFLOP per image for
# gemma4v) on EVERY backend — the CUDA targets compile media.c as C too, and
# without this they run it single-threaded (minutes per image, found the hard way).
find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(run PRIVATE OpenMP::OpenMP_C)
endif()

# --- CUDA build (optional) -------------------------------------------------
# Ampere (sm_80) is the floor — build for Ampere and forward. fa_mma uses mma.m16n8k16.f16
# and the staging kernels use cp.async, both sm_80+; a sub-Ampere build won't compile.
set(LG_CUDA_ARCH "86" CACHE STRING "CUDA architecture, Ampere+ (A5000 = 86, Orin NX = 87)")
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    # Two CUDA backends behind the same model.h, sharing model-cuda.cuh:
    # run-cuda = readable f32-dequant matmul; run-cuda-i8 = int8 matmul with
    # wide loads from (re)aligned weights. (An earlier byte-load int8 variant
    # lives in git history — see the README.)
    # media-kernel.cu: the gemma4v encoder on the GPU (~1 TFLOP/image — ~15 s on
    # a good CPU, well under a second on the GPU). The host encoder stays in as
    # oracle and fallback. media.c calls the GPU seam unconditionally; the CPU run
    # target links a no-op stub (media-gpu-stub.c), the CUDA targets link media-kernel.cu.
    add_executable(run-cuda     ${CORE} src/cuda/model-cuda-f32.cu src/cuda/media-kernel.cu)
    add_executable(run-cuda-i8  ${CORE} src/cuda/model-cuda-i8.cu src/cuda/media-kernel.cu)
    foreach(tgt run-cuda run-cuda-i8)
        target_include_directories(${tgt} PRIVATE include src)   # src/ for media-internal.h & mtp-internal.h
        if(WIN32)
            target_link_libraries(${tgt} PRIVATE ws2_32)   # AF_UNIX sockets (-s/-c)
            # NVIDIA's prebuilt cudart_static.lib uses the STATIC CRT (LIBCMT[D]); our code
            # uses the DYNAMIC CRT (MSVCRT). Drop the redundant static one so the binary
            # resolves a single CRT (silences LNK4098). MSVC-only; a no-op elsewhere.
            target_link_options(${tgt} PRIVATE "/NODEFAULTLIB:$<IF:$<CONFIG:Debug>,LIBCMTD,LIBCMT>")
        elseif(UNIX)
            target_link_libraries(${tgt} PRIVATE m)
        endif()
        if(OpenMP_C_FOUND)
            target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_C)
        endif()
        # dev box A5000 = sm_86; Orin NX: cmake -DLG_CUDA_ARCH=87
        set_target_properties(${tgt} PROPERTIES CUDA_ARCHITECTURES "${LG_CUDA_ARCH}")
        # per-thread default stream: required to stream-capture the forward into a CUDA graph
        # (the legacy default stream is not capturable). model_forward always replays a graph.
        target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
    endforeach()
    message(STATUS "CUDA found (${CMAKE_CUDA_COMPILER_VERSION}) — building run-cuda, run-cuda-i8")
else()
    message(STATUS "CUDA not found — skipping run-cuda")
endif()

# --- tools -------------------------------------------------------------------
# socket_cat: AF_UNIX stdin/stdout client for talking to `run -s` on systems
# without socat (native Windows, most of all).
add_executable(socket_cat tools/socket_cat.c)
if(WIN32)
    target_link_libraries(socket_cat PRIVATE ws2_32)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(socket_cat PRIVATE Threads::Threads)
endif()

# media files (decode/resize/resample) are handled by mmcat in the sibling repo
# little-gemma-tools — kept out of the core so it stays dependency-free C/CUDA.

# --- tensor/graph self-test ------------------------------------------------
add_executable(graph_test test/graph_test.c src/graph.c)
target_include_directories(graph_test PRIVATE include)
if(UNIX)
    target_link_libraries(graph_test PRIVATE m)
endif()

enable_testing()
add_test(NAME graph_test COMMAND graph_test)
