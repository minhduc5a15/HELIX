#include <algorithm>

#include "../simd/simd_avx2.hpp"
#include "matmul_config.hpp"
#include "matmul_kernel.hpp"

namespace helix {

#if defined(__AVX2__)

    // 4x16 Outer Product Micro-Kernel
    // Computes a 4x16 block of C using 4 rows of A and 16 cols of B.
    // Keeps C in 8 YMM registers for the entire K loop.
    static inline void micro_kernel_4x16(
        const float* a, const float* b, float* out, size_t i, size_t j, size_t K, size_t stride_N, size_t stride_K
    ) {
        __m256 c00 = simd::setzero();
        __m256 c01 = simd::setzero();
        __m256 c10 = simd::setzero();
        __m256 c11 = simd::setzero();
        __m256 c20 = simd::setzero();
        __m256 c21 = simd::setzero();
        __m256 c30 = simd::setzero();
        __m256 c31 = simd::setzero();

        for (size_t k = 0; k < K; ++k) {
            __m256 b0 = simd::load(&b[k * stride_N + j]);
            __m256 b1 = simd::load(&b[k * stride_N + j + 8]);

            __m256 a0 = simd::broadcast(a[(i + 0) * stride_K + k]);
            c00 = simd::fmadd(a0, b0, c00);
            c01 = simd::fmadd(a0, b1, c01);

            __m256 a1 = simd::broadcast(a[(i + 1) * stride_K + k]);
            c10 = simd::fmadd(a1, b0, c10);
            c11 = simd::fmadd(a1, b1, c11);

            __m256 a2 = simd::broadcast(a[(i + 2) * stride_K + k]);
            c20 = simd::fmadd(a2, b0, c20);
            c21 = simd::fmadd(a2, b1, c21);

            __m256 a3 = simd::broadcast(a[(i + 3) * stride_K + k]);
            c30 = simd::fmadd(a3, b0, c30);
            c31 = simd::fmadd(a3, b1, c31);
        }

        simd::store(&out[(i + 0) * stride_N + j], c00);
        simd::store(&out[(i + 0) * stride_N + j + 8], c01);
        simd::store(&out[(i + 1) * stride_N + j], c10);
        simd::store(&out[(i + 1) * stride_N + j + 8], c11);
        simd::store(&out[(i + 2) * stride_N + j], c20);
        simd::store(&out[(i + 2) * stride_N + j + 8], c21);
        simd::store(&out[(i + 3) * stride_N + j], c30);
        simd::store(&out[(i + 3) * stride_N + j + 8], c31);
    }

    // 4x8 Micro-Kernel for N-tail
    static inline void micro_kernel_4x8(
        const float* a, const float* b, float* out, size_t i, size_t j, size_t K, size_t stride_N, size_t stride_K
    ) {
        __m256 c00 = simd::setzero();
        __m256 c10 = simd::setzero();
        __m256 c20 = simd::setzero();
        __m256 c30 = simd::setzero();

        for (size_t k = 0; k < K; ++k) {
            __m256 b0 = simd::load(&b[k * stride_N + j]);

            __m256 a0 = simd::broadcast(a[(i + 0) * stride_K + k]);
            c00 = simd::fmadd(a0, b0, c00);

            __m256 a1 = simd::broadcast(a[(i + 1) * stride_K + k]);
            c10 = simd::fmadd(a1, b0, c10);

            __m256 a2 = simd::broadcast(a[(i + 2) * stride_K + k]);
            c20 = simd::fmadd(a2, b0, c20);

            __m256 a3 = simd::broadcast(a[(i + 3) * stride_K + k]);
            c30 = simd::fmadd(a3, b0, c30);
        }

        simd::store(&out[(i + 0) * stride_N + j], c00);
        simd::store(&out[(i + 1) * stride_N + j], c10);
        simd::store(&out[(i + 2) * stride_N + j], c20);
        simd::store(&out[(i + 3) * stride_N + j], c30);
    }
#endif

    // Processes a block (ih to ih+M_block, jh to jh+N_block) for full K
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
    ) {
#if defined(__AVX2__)
        size_t i = ih;
        const size_t i_end = ih + M_block;
        const size_t j_end = jh + N_block;

        for (; i + 3 < i_end; i += 4) {
            size_t j = jh;
            for (; j + 15 < j_end; j += 16) {
                micro_kernel_4x16(a, b, out, i, j, K, stride_N, stride_K);
            }
            for (; j + 7 < j_end; j += 8) {
                micro_kernel_4x8(a, b, out, i, j, K, stride_N, stride_K);
            }
            // N scalar tail
            for (; j < j_end; ++j) {
                float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                for (size_t k = 0; k < K; ++k) {
                    float bk = b[k * stride_N + j];
                    c0 += a[(i + 0) * stride_K + k] * bk;
                    c1 += a[(i + 1) * stride_K + k] * bk;
                    c2 += a[(i + 2) * stride_K + k] * bk;
                    c3 += a[(i + 3) * stride_K + k] * bk;
                }
                out[(i + 0) * stride_N + j] = c0;
                out[(i + 1) * stride_N + j] = c1;
                out[(i + 2) * stride_N + j] = c2;
                out[(i + 3) * stride_N + j] = c3;
            }
        }

        // M scalar tail
        for (; i < i_end; ++i) {
            for (size_t j = jh; j < j_end; ++j) {
                float c = 0;
                for (size_t k = 0; k < K; ++k) {
                    c += a[i * stride_K + k] * b[k * stride_N + j];
                }
                out[i * stride_N + j] = c;
            }
        }
#else
        // Fallback for non-AVX2
        for (size_t i = ih; i < ih + M_block; ++i) {
            for (size_t j = jh; j < jh + N_block; ++j) {
                float c = 0;
                for (size_t k = 0; k < K; ++k) {
                    c += a[i * stride_K + k] * b[k * stride_N + j];
                }
                out[i * stride_N + j] = c;
            }
        }
#endif
    }

    void avx2_micro_matmul(const float* a, const float* b, float* out, size_t M, size_t K, size_t N) {
        constexpr size_t BLOCK = MatMulConfig::block_size;
        for (size_t ih = 0; ih < M; ih += BLOCK) {
            size_t i_end = std::min(ih + BLOCK, M);
            for (size_t jh = 0; jh < N; jh += BLOCK) {
                size_t j_end = std::min(jh + BLOCK, N);
                avx2_matmul_block(a, b, out, i_end - ih, K, j_end - jh, ih, jh, N, K);
            }
        }
    }
}  // namespace helix
