cmake_minimum_required(VERSION 3.25)
project(helix VERSION 1.0.0 LANGUAGES CXX)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add include directory
include_directories(include)

# Build options
option(HELIX_NATIVE_BUILD "Enable -march=native and LTO for local performance builds" OFF)
option(HELIX_ENABLE_TSAN "Enable ThreadSanitizer (Requires Clang/LLVM)" OFF)

if(HELIX_ENABLE_TSAN)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "ThreadSanitizer in HELIX requires Clang/LLVM to avoid false positives with OpenMP barriers. Please run build with CC=clang CXX=clang++.")
    endif()
    add_compile_options(-fsanitize=thread -g -fPIE)
    add_link_options(-fsanitize=thread -g -fPIE -pie)
endif()

# Define the HELIX core library
add_library(helix 
    src/core/allocator.cpp
    src/core/storage.cpp
    src/core/tensor.cpp
    src/core/tensor_factory.cpp
    src/core/dispatcher.cpp
    src/core/autotuner.cpp
    src/core/broadcast.cpp
    src/backend/kernels/ops.cpp
    src/backend/kernels/reduce.cpp
    src/backend/kernels/loss.cpp
    src/backend/matmul/naive_matmul.cpp
    src/backend/matmul/blocked_matmul.cpp
    src/backend/matmul/avx2_dot_matmul.cpp
    src/backend/matmul/avx2_micro_matmul.cpp
    src/backend/matmul/openmp_matmul.cpp
    src/backend/matmul/matmul_backend.cpp
    src/backend/simd/simd_backend.cpp
    src/autograd/autograd_meta.cpp
    src/autograd/function.cpp
    src/autograd/graph_builder.cpp
    src/autograd/engine.cpp
    src/nn/module.cpp
    src/nn/linear.cpp
    src/nn/relu.cpp
    src/nn/sequential.cpp
    src/nn/loss.cpp
    src/optim/sgd.cpp
    src/benchmark/benchmark_runner.cpp
    src/benchmark/benchmark_reporter.cpp
)
target_include_directories(helix PUBLIC include)

# Compiler Warnings
if(MSVC)
    target_compile_options(helix PRIVATE /W4)
else()
    target_compile_options(helix PRIVATE -Wall -Wextra -Wpedantic)
endif()

# OpenMP Configuration
find_package(OpenMP)
if(OpenMP_FOUND)
    target_link_libraries(helix PUBLIC OpenMP::OpenMP_CXX)
    if(MSVC)
        target_compile_options(helix PUBLIC -openmp:experimental)
    endif()
endif()

# Compiler Flags (AVX2/FMA only on x86_64)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "amd64|AMD64|x86_64|X86_64")
    if(MSVC)
        target_compile_options(helix PUBLIC /arch:AVX2 /O2)
    else()
        if(HELIX_NATIVE_BUILD)
            target_compile_options(helix PUBLIC -march=native -O3 -flto)
        else()
            target_compile_options(helix PUBLIC -mavx2 -mfma -O3)
        endif()
    endif()
else()
    if(MSVC)
        target_compile_options(helix PUBLIC /O2)
    else()
        target_compile_options(helix PUBLIC -O3)
    endif()
endif()

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        03597a01ee50ed33e9dfd640b249b4be3799d395
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

add_subdirectory(tests)
add_subdirectory(stress)
add_subdirectory(benchmark)
add_subdirectory(examples)
