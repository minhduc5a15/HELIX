#pragma once

#include <cstddef>

namespace helix {

    class CPUBackend {
    public:
        // Core element-wise operations
        // Assumes all inputs and outputs are contiguous in memory and have the same number of elements.
        static void add(const float* a, const float* b, float* out, size_t size);
        static void sub(const float* a, const float* b, float* out, size_t size);
        static void mul(const float* a, const float* b, float* out, size_t size);
        static void div(const float* a, const float* b, float* out, size_t size);

        // Unary operations
        static void neg(const float* a, float* out, size_t size);

        // Matrix Multiplication
        static void matmul(const float* a, const float* b_t, float* out, size_t M, size_t K, size_t N);
    };

}  // namespace helix
