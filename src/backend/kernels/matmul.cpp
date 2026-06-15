#include "backend/cpu_backend.hpp"

namespace helix {

    void CPUBackend::matmul(const float* a, const float* b_t, float* out, size_t M, size_t K, size_t N) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    // a is M x K
                    // b_t is N x K (meaning each row is length K)
                    sum += a[i * K + k] * b_t[j * K + k];
                }
                out[i * N + j] = sum;
            }
        }
    }

}  // namespace helix
