#include <algorithm>
#include <cstddef>  // ptrdiff_t

#include "core/cpu_utils.hpp"
#include "matmul_config.hpp"
#include "matmul_kernel.hpp"

#if defined(__AVX2__)
#include <immintrin.h>
#if defined(__FMA__) || (defined(_MSC_VER) && defined(__AVX2__))
#define HELIX_USE_FMA
#endif
#endif

namespace helix {
    inline bool supports_avx2_internal() {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
        return cpu_supports_avx2_fma();
#else
        return true;
#endif
#else
        return false;
#endif
    }

    void openmp_matmul(const float* a, const float* b, float* out, const size_t M, const size_t K, const size_t N) {
        if (M == 0 || N == 0) {
            return;
        }
        if (K == 0) {
            std::fill_n(out, M * N, 0.0f);
            return;
        }

        const bool use_avx2 = supports_avx2_internal();
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
        // Switch to ptrdiff_t for MSVC compatibility and to prevent overflow
        const ptrdiff_t M_s = static_cast<ptrdiff_t>(M);
        const ptrdiff_t N_s = static_cast<ptrdiff_t>(N);
        const ptrdiff_t BLOCK_s = static_cast<ptrdiff_t>(BLOCK);

        // MSVC does not support collapse with unsigned types, but collapse(2) works with signed types.
        // However, it logs a "collapse clause ignored" warning; we can either drop collapse
        // or keep it (it still runs correctly, just loses optimization). Keeping it for compatibility.
#if defined(_MSC_VER)
#pragma omp parallel for schedule(dynamic)
#else
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
        for (ptrdiff_t ih = 0; ih < M_s; ih += BLOCK_s) {
#if defined(_MSC_VER)
            // If collapse is not used, an outer loop for the parallel for would be needed,
            // but since we used collapse, MSVC will ignore the collapse clause while still creating
            // the parallel for with the signed loop variable ih, which is fine.
#endif
            for (ptrdiff_t jh = 0; jh < N_s; jh += BLOCK_s) {
                const size_t i_begin = static_cast<size_t>(ih);
                const size_t j_begin = static_cast<size_t>(jh);
                const size_t i_end = std::min(static_cast<size_t>(ih + BLOCK_s), M);
                const size_t j_end = std::min(static_cast<size_t>(jh + BLOCK_s), N);

                if (use_avx2) {
                    // AVX2 kernel overwrites directly, no initialization required
                    avx2_matmul_block(a, b, out, i_end - i_begin, K, j_end - j_begin, i_begin, j_begin, N, K);
                } else {
                    // Fallback: needs element-wise zero initialization before accumulation
                    for (size_t kh = 0; kh < K; kh += BLOCK) {
                        const size_t k_end = std::min(kh + BLOCK, K);

                        for (size_t i = i_begin; i < i_end; ++i) {
                            for (size_t j = j_begin; j < j_end; ++j) {
                                if (kh == 0) {
                                    out[i * N + j] = 0.0f;
                                }
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
        // Non-OpenMP fallback
        if (use_avx2) {
            avx2_micro_matmul(a, b, out, M, K, N);
        } else {
            blocked_matmul(a, b, out, M, K, N);
        }
#endif
    }
}  // namespace helix
