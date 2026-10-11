#pragma once

#include <cstddef>
#include <cstdint>

#include "backend/cpu_backend.hpp"

namespace helix::internal {

    template <typename T>
    void bmm_serial_batches(
        const T* a, const T* b, T* out, size_t B, size_t M, size_t K, size_t N, MatMulStrategy strategy
    );

    void bmm_parallel_batches_float(const float* a, const float* b, float* out, size_t B, size_t M, size_t K, size_t N);

}  // namespace helix::internal
