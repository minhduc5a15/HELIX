#include "matmul_kernel.hpp"
#include "matmul_config.hpp"
#include <algorithm>

namespace helix {
    void blocked_matmul(const float* a, const float* b_t, float* out, size_t M, size_t K, size_t N) {
        std::fill_n(out, M * N, 0.0f);

        constexpr size_t BLOCK = MatMulConfig::block_size;

        for (size_t ih = 0; ih < M; ih += BLOCK) {
            const size_t i_end = std::min(ih + BLOCK, M);
            for (size_t jh = 0; jh < N; jh += BLOCK) {
                const size_t j_end = std::min(jh + BLOCK, N);
                for (size_t kh = 0; kh < K; kh += BLOCK) {
                    const size_t k_end = std::min(kh + BLOCK, K);

                    for (size_t i = ih; i < i_end; ++i) {
                        for (size_t j = jh; j < j_end; ++j) {
                            float sum = 0.0f;
                            for (size_t k = kh; k < k_end; ++k) {
                                sum += a[i * K + k] * b_t[j * K + k];
                            }
                            out[i * N + j] += sum;
                        }
                    }
                }
            }
        }
    }
}
