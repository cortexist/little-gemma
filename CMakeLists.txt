cmake_minimum_required(VERSION 3.15)

project(little-gemma C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Common warning flags applied to every target.
if(MSVC)
    add_compile_options(/W3)
else()
    add_compile_options(-Wall -Wextra)
endif()

# The GGUF inspector.
add_executable(run src/run.c src/gguf.c src/quant.c src/model.c src/tokenizer.c)
target_include_directories(run PRIVATE include)

# Multi-threaded matmul via OpenMP (optional).
find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(run PRIVATE OpenMP::OpenMP_C)
endif()

# The tensor/graph layer self-test.
add_executable(graph_test test/graph_test.c src/graph.c)
target_include_directories(graph_test PRIVATE include)

# Link the math library on Unix-like systems (not needed/available with MSVC).
if(UNIX)
    target_link_libraries(graph_test PRIVATE m)
endif()

enable_testing()
add_test(NAME graph_test COMMAND graph_test)
