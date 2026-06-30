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
    // Helper to check AVX2 support at runtime
    inline bool supports_avx2_internal() {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
        return true;  
#endif
#else
        return false; 
#endif
    }

    void openmp_matmul(const float* a, const float* b_t, float* out, size_t M, size_t K, size_t N) {
        bool use_avx2 = supports_avx2_internal();
#if defined(_OPENMP)
        std::fill_n(out, M * N, 0.0f);
        constexpr size_t BLOCK = MatMulConfig::block_size;

#pragma omp parallel for collapse(2) schedule(static)
        for (int ih = 0; ih < static_cast<int>(M); ih += static_cast<int>(BLOCK)) {
            for (int jh = 0; jh < static_cast<int>(N); jh += static_cast<int>(BLOCK)) {
                const size_t i_end = std::min(static_cast<size_t>(ih) + BLOCK, M);
                const size_t j_end = std::min(static_cast<size_t>(jh) + BLOCK, N);

                for (size_t kh = 0; kh < K; kh += BLOCK) {
                    const size_t k_end = std::min(kh + BLOCK, K);

                    for (size_t i = static_cast<size_t>(ih); i < i_end; ++i) {
                        for (size_t j = static_cast<size_t>(jh); j < j_end; ++j) {
                            float sum = 0.0f;

                            if (use_avx2) {
#if defined(__AVX2__)
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
                                sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

                                for (; k < k_end; ++k) {
                                    sum += a[i * K + k] * b_t[j * K + k];
                                }
#endif
                            } else {
                                for (size_t k = kh; k < k_end; ++k) {
                                    sum += a[i * K + k] * b_t[j * K + k];
                                }
                            }
                            out[i * N + j] += sum;
                        }
                    }
                }
            }
        }
#else
        if (use_avx2) {
            avx2_matmul(a, b_t, out, M, K, N);
        } else {
            blocked_matmul(a, b_t, out, M, K, N);
        }
#endif
    }
}
