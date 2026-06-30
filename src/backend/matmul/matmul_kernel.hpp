#pragma once
#include <cstddef>

namespace helix {
    void naive_matmul(const float* a, const float* b_t, float* out, size_t M, size_t K, size_t N);
    void blocked_matmul(const float* a, const float* b_t, float* out, size_t M, size_t K, size_t N);
    void avx2_dot_matmul(const float* a, const float* b_t, float* out, size_t M, size_t K, size_t N);
    void avx2_micro_matmul(const float* a, const float* b_t, float* out, size_t M, size_t K, size_t N);
    void openmp_matmul(const float* a, const float* b_t, float* out, size_t M, size_t K, size_t N);
}  // namespace helix
