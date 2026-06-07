cmake_minimum_required(VERSION 3.15)

project(little-gemma C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(run src/run.c src/gguf.c)

target_include_directories(run PRIVATE include)

# Link the math library on Unix-like systems (not needed/available with MSVC).
if(UNIX)
    target_link_libraries(run PRIVATE m)
endif()

# Enable common warnings.
if(MSVC)
    target_compile_options(run PRIVATE /W3)
else()
    target_compile_options(run PRIVATE -Wall -Wextra)
endif()
