#pragma once

#include <cstddef>

namespace helix {

    class SIMDBackend {
    public:
        // Returns true if SIMD backend supports the current platform
        static bool is_supported();

        static void add(const float* a, const float* b, float* out, size_t size);
        static void sub(const float* a, const float* b, float* out, size_t size);
        static void mul(const float* a, const float* b, float* out, size_t size);
        static void div(const float* a, const float* b, float* out, size_t size);

        static void add_scalar(const float* a, float scalar, float* out, size_t size);
        static void sub_scalar(const float* a, float scalar, float* out, size_t size);
        static void mul_scalar(const float* a, float scalar, float* out, size_t size);
        static void div_scalar(const float* a, float scalar, float* out, size_t size);

        static void neg(const float* a, float* out, size_t size);
        static void relu(const float* a, float* out, size_t size);

        static void sum(const float* input, float* output, size_t outer_size, size_t dim_size, size_t inner_size);
        static void mean(const float* input, float* output, size_t outer_size, size_t dim_size, size_t inner_size);
    };

}  // namespace helix
