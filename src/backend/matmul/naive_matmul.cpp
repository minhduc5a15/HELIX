#include "matmul_kernel.hpp"

namespace helix {
    void naive_matmul(const float* a, const float* b, float* out, const size_t M, const size_t K, const size_t N) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += a[i * K + k] * b[k * N + j];
                }
                out[i * N + j] = sum;
            }
        }
    }
}  // namespace helix
