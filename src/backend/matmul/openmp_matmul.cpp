#include <algorithm>

#include "matmul_config.hpp"
#include "matmul_kernel.hpp"

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

    void openmp_matmul(const float* a, const float* b, float* out, size_t M, size_t K, size_t N) {
        bool use_avx2 = supports_avx2_internal();
        constexpr size_t BLOCK = MatMulConfig::block_size;

        // Bypass OpenMP thread management overhead for small matrices
        if (M <= BLOCK && N <= BLOCK) {
            if (use_avx2) {
                avx2_micro_matmul(a, b, out, M, K, N);
            } else {
                blocked_matmul(a, b, out, M, K, N);
            }
            return;
        }

#if defined(_OPENMP)
        std::fill_n(out, M * N, 0.0f);

#pragma omp parallel for collapse(2) schedule(dynamic)
        for (int ih = 0; ih < static_cast<int>(M); ih += static_cast<int>(BLOCK)) {
            for (int jh = 0; jh < static_cast<int>(N); jh += static_cast<int>(BLOCK)) {
                const size_t i_end = std::min(static_cast<size_t>(ih) + BLOCK, M);
                const size_t j_end = std::min(static_cast<size_t>(jh) + BLOCK, N);

                if (use_avx2) {
                    // For AVX2, we delegate to the specialized micro-kernel block
                    avx2_matmul_block(a, b, out, i_end - ih, K, j_end - jh, ih, jh, N, K);
                } else {
                    for (size_t kh = 0; kh < K; kh += BLOCK) {
                        const size_t k_end = std::min(kh + BLOCK, K);

                        for (size_t i = static_cast<size_t>(ih); i < i_end; ++i) {
                            for (size_t j = static_cast<size_t>(jh); j < j_end; ++j) {
                                float sum = 0.0f;
                                for (size_t k = kh; k < k_end; ++k) {
                                    sum += a[i * K + k] * b[k * N + j];
                                }
                                out[i * N + j] += sum;
                            }
                        }
                    }
                }
            }
        }
#else
        if (use_avx2) {
            avx2_micro_matmul(a, b, out, M, K, N);
        } else {
            blocked_matmul(a, b, out, M, K, N);
        }
#endif
    }
}  // namespace helix
