#include "matmul_kernel.hpp"
#include "matmul_config.hpp"
#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#if defined(__FMA__) || (defined(_MSC_VER) && defined(__AVX2__))
#define HELIX_USE_FMA
#endif
#endif

namespace helix {
    void avx2_matmul(const float* a, const float* b_t, float* out, size_t M, size_t K, size_t N) {
        std::fill_n(out, M * N, 0.0f);

#if defined(__AVX2__)
        constexpr size_t BLOCK = MatMulConfig::block_size;

        for (size_t ih = 0; ih < M; ih += BLOCK) {
            const size_t i_end = std::min(ih + BLOCK, M);
            for (size_t jh = 0; jh < N; jh += BLOCK) {
                const size_t j_end = std::min(jh + BLOCK, N);
                for (size_t kh = 0; kh < K; kh += BLOCK) {
                    const size_t k_end = std::min(kh + BLOCK, K);

                    for (size_t i = ih; i < i_end; ++i) {
                        for (size_t j = jh; j < j_end; ++j) {
                            __m256 acc = _mm256_setzero_ps();
                            size_t k = kh;
                            for (; k + 7 < k_end; k += 8) {
                                const __m256 va = _mm256_loadu_ps(&a[i * K + k]);
                                const __m256 vb = _mm256_loadu_ps(&b_t[j * K + k]);
#if defined(HELIX_USE_FMA)
                                acc = _mm256_fmadd_ps(va, vb, acc);
#else
                                acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
#endif
                            }

                            alignas(32) float temp[8];
                            _mm256_storeu_ps(temp, acc);
                            float sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

                            for (; k < k_end; ++k) {
                                sum += a[i * K + k] * b_t[j * K + k];
                            }
                            out[i * N + j] += sum;
                        }
                    }
                }
            }
        }
#else
        blocked_matmul(a, b_t, out, M, K, N);
#endif
    }
}
