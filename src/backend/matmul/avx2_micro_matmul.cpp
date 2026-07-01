#include <algorithm>

#include "../simd/simd_avx2.hpp"
#include "matmul_config.hpp"
#include "matmul_kernel.hpp"

namespace helix {

#if defined(__AVX2__)
    // WHY AN 8x1 MICRO-KERNEL?
    // This micro-kernel computes an 8-element column of matrix C (from i to i+7).
    // By keeping 8 YMM registers (c0 to c7) to accumulate intermediate results,
    // we only need to load 1 vector from B_T (vb) and 8 vectors from A to perform 8 FMAs.
    // Arithmetic Intensity = 8 FMAs / 9 Loads = ~0.88.
    // This is the simplest design to utilize AVX2 but not yet fully optimal.
    // To achieve higher GFLOPS, Register Blocking (e.g. 4x3 or 3x4) is required (~1.7 FMA/Load).
    static inline void micro_kernel_8x1(
        const float* a, const float* b_t, float* out, size_t i, size_t j, size_t kh, size_t k_end, size_t K, size_t N
    ) {
        __m256 c0 = simd::setzero();
        __m256 c1 = simd::setzero();
        __m256 c2 = simd::setzero();
        __m256 c3 = simd::setzero();
        __m256 c4 = simd::setzero();
        __m256 c5 = simd::setzero();
        __m256 c6 = simd::setzero();
        __m256 c7 = simd::setzero();

        size_t k = kh;
        // Process K dimension in chunks of 8
        for (; k + 7 < k_end; k += 8) {
            __m256 vb = simd::load(&b_t[j * K + k]);
            c0 = simd::fmadd(simd::load(&a[(i + 0) * K + k]), vb, c0);
            c1 = simd::fmadd(simd::load(&a[(i + 1) * K + k]), vb, c1);
            c2 = simd::fmadd(simd::load(&a[(i + 2) * K + k]), vb, c2);
            c3 = simd::fmadd(simd::load(&a[(i + 3) * K + k]), vb, c3);
            c4 = simd::fmadd(simd::load(&a[(i + 4) * K + k]), vb, c4);
            c5 = simd::fmadd(simd::load(&a[(i + 5) * K + k]), vb, c5);
            c6 = simd::fmadd(simd::load(&a[(i + 6) * K + k]), vb, c6);
            c7 = simd::fmadd(simd::load(&a[(i + 7) * K + k]), vb, c7);
        }

        // Horizontal sum and scalar tail for K
        auto hsum_and_tail = [&](__m256 acc, size_t row) {
            alignas(32) float temp[8];
            simd::store(temp, acc);
            float sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

            // K-tail (handles K not divisible by 8)
            for (size_t tk = k; tk < k_end; ++tk) {
                sum += a[row * K + tk] * b_t[j * K + tk];
            }
            out[row * N + j] += sum;
        };

        hsum_and_tail(c0, i + 0);
        hsum_and_tail(c1, i + 1);
        hsum_and_tail(c2, i + 2);
        hsum_and_tail(c3, i + 3);
        hsum_and_tail(c4, i + 4);
        hsum_and_tail(c5, i + 5);
        hsum_and_tail(c6, i + 6);
        hsum_and_tail(c7, i + 7);
    }
#endif

    void avx2_micro_matmul(const float* a, const float* b_t, float* out, size_t M, size_t K, size_t N) {
        std::fill_n(out, M * N, 0.0f);

#if defined(__AVX2__)
        constexpr size_t BLOCK = MatMulConfig::block_size;

        for (size_t ih = 0; ih < M; ih += BLOCK) {
            const size_t i_end = std::min(ih + BLOCK, M);
            for (size_t jh = 0; jh < N; jh += BLOCK) {
                const size_t j_end = std::min(jh + BLOCK, N);
                for (size_t kh = 0; kh < K; kh += BLOCK) {
                    const size_t k_end = std::min(kh + BLOCK, K);

                    size_t i = ih;
                    // 8x1 Micro-kernel loop
                    for (; i + 7 < i_end; i += 8) {
                        // For N dimension, we process element by element (N-tail is naturally handled)
                        for (size_t j = jh; j < j_end; ++j) {
                            micro_kernel_8x1(a, b_t, out, i, j, kh, k_end, K, N);
                        }
                    }

                    // Tail handling for M dimension (M not divisible by 8)
                    for (; i < i_end; ++i) {
                        for (size_t j = jh; j < j_end; ++j) {
                            float sum = 0.0f;
                            size_t k = kh;

                            // Vectorized K for M-tail
                            __m256 v_sum = simd::setzero();
                            for (; k + 7 < k_end; k += 8) {
                                __m256 va = simd::load(&a[i * K + k]);
                                __m256 vb = simd::load(&b_t[j * K + k]);
                                v_sum = simd::fmadd(va, vb, v_sum);
                            }
                            alignas(32) float temp[8];
                            simd::store(temp, v_sum);
                            sum += temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

                            // Scalar K-tail for M-tail
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
}  // namespace helix
