cmake_minimum_required(VERSION 3.18)

project(little-gemma C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Warning flags for C only (so they don't reach nvcc).
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:C>:/W3>)
else()
    add_compile_options($<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra>)
endif()

# Shared, backend-agnostic sources. model.c = common host setup; the compute
# backend is model-cpu.c (CPU) or model-cuda-{f32,i8}.cu (CUDA), selected per target.
set(CORE src/run.c src/gguf.c src/quant.c src/model.c src/tokenizer.c)

# --- CPU build -------------------------------------------------------------
add_executable(run ${CORE} src/model-cpu.c)
target_include_directories(run PRIVATE include)

find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(run PRIVATE OpenMP::OpenMP_C)
endif()

# --- CUDA build (optional) -------------------------------------------------
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    # Two CUDA backends behind the same model.h, sharing model-cuda.cuh:
    # run-cuda = readable f32-dequant matmul; run-cuda-i8r = int8 matmul with
    # wide loads from (re)aligned weights. (The intermediate byte-load int8
    # backend, model-cuda-i8.cu, lives in git history — see the README.)
    add_executable(run-cuda     ${CORE} src/model-cuda-f32.cu)
    add_executable(run-cuda-i8r ${CORE} src/model-cuda-i8r.cu)
    foreach(tgt run-cuda run-cuda-i8r)
        target_include_directories(${tgt} PRIVATE include)
        set_target_properties(${tgt} PROPERTIES CUDA_ARCHITECTURES "86")
        # per-thread default stream: required to stream-capture the forward into a CUDA graph
        # (the legacy default stream is not capturable). model_forward always replays a graph.
        target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
    endforeach()
    message(STATUS "CUDA found (${CMAKE_CUDA_COMPILER_VERSION}) — building run-cuda, run-cuda-i8r")
else()
    message(STATUS "CUDA not found — skipping run-cuda")
endif()

# --- tensor/graph self-test ------------------------------------------------
add_executable(graph_test test/graph_test.c src/graph.c)
target_include_directories(graph_test PRIVATE include)
if(UNIX)
    target_link_libraries(graph_test PRIVATE m)
endif()

enable_testing()
add_test(NAME graph_test COMMAND graph_test)
