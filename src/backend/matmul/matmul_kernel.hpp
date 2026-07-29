#pragma once
#include <cstddef>

namespace helix {
    void naive_matmul(const float* a, const float* b, float* out, size_t M, size_t K, size_t N);
    void blocked_matmul(const float* a, const float* b, float* out, size_t M, size_t K, size_t N);
    void avx2_dot_matmul(const float* a, const float* b, float* out, size_t M, size_t K, size_t N);
    void avx2_micro_matmul(const float* a, const float* b, float* out, size_t M, size_t K, size_t N);
    void avx2_matmul_block(
        const float* a,
        const float* b,
        float* out,
        size_t M_block,
        size_t K,
        size_t N_block,
        size_t ih,
        size_t jh,
        size_t stride_N,
        size_t stride_K
    );
    void openmp_matmul(const float* a, const float* b, float* out, size_t M, size_t K, size_t N);
}  // namespace helix
