#include <algorithm>

#include "backend/cpu_backend.hpp"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(_OPENMP)
#endif

namespace helix {

#if defined(__AVX2__)
#if defined(__FMA__) || (defined(_MSC_VER) && defined(__AVX2__))
#define HELIX_USE_FMA
#endif
#endif

    void CPUBackend::matmul_naive(
        const float* a, const float* b_t, float* out, const size_t M, const size_t K, const size_t N
    ) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += a[i * K + k] * b_t[j * K + k];
                }
                out[i * N + j] = sum;
            }
        }
    }

    template <size_t TileM = 64, size_t TileN = 64, size_t TileK = 64>
    void kernel_matmul_tiled(
        const float* a, const float* b_t, float* out, const size_t M, const size_t K, const size_t N
    ) {
        std::fill_n(out, M * N, 0.0f);

        for (size_t ih = 0; ih < M; ih += TileM) {
            const size_t i_end = std::min(ih + TileM, M);
            for (size_t jh = 0; jh < N; jh += TileN) {
                const size_t j_end = std::min(jh + TileN, N);
                for (size_t kh = 0; kh < K; kh += TileK) {
                    const size_t k_end = std::min(kh + TileK, K);

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

    void CPUBackend::matmul_tiled(
        const float* a, const float* b_t, float* out, const size_t M, const size_t K, const size_t N
    ) {
        // Tile size can be modified via template arguments
        kernel_matmul_tiled<64, 64, 64>(a, b_t, out, M, K, N);
    }

    void CPUBackend::matmul_avx2(
        const float* a, const float* b_t, float* out, const size_t M, const size_t K, const size_t N
    ) {
        std::fill_n(out, M * N, 0.0f);

#if defined(__AVX2__)
        constexpr size_t TileM = 64;
        constexpr size_t TileN = 64;
        constexpr size_t TileK = 64;

        for (size_t ih = 0; ih < M; ih += TileM) {
            const size_t i_end = std::min(ih + TileM, M);
            for (size_t jh = 0; jh < N; jh += TileN) {
                const size_t j_end = std::min(jh + TileN, N);
                for (size_t kh = 0; kh < K; kh += TileK) {
                    const size_t k_end = std::min(kh + TileK, K);

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
        matmul_tiled(a, b_t, out, M, K, N);
#endif
    }

    // Helper to check AVX2 support at runtime
    inline bool supports_avx2() {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
        return true; // Assume true or implement __cpuid for MSVC
#endif
#else
        return false; // ARM64 and others do not support AVX2
#endif
    }

    void CPUBackend::matmul(
        const float* a, const float* b_t, float* out, const size_t M, const size_t K, const size_t N
    ) {
        bool use_avx2 = supports_avx2();

        constexpr size_t threshold = 16384;  // 128 * 128 threshold to trigger multithreading
        if (M * N >= threshold) {
#if defined(_OPENMP)
            std::fill_n(out, M * N, 0.0f);
            constexpr size_t TileM = 64;
            constexpr size_t TileN = 64;
            constexpr size_t TileK = 64;

#pragma omp parallel for collapse(2) schedule(static)
            for (int ih = 0; ih < static_cast<int>(M); ih += static_cast<int>(TileM)) {
                for (int jh = 0; jh < static_cast<int>(N); jh += static_cast<int>(TileN)) {
                    const size_t i_end = std::min(static_cast<size_t>(ih) + TileM, M);
                    const size_t j_end = std::min(static_cast<size_t>(jh) + TileN, N);

                    for (size_t kh = 0; kh < K; kh += TileK) {
                        const size_t k_end = std::min(kh + TileK, K);

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
                                    // Fallback to scalar
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
                matmul_avx2(a, b_t, out, M, K, N);
            } else {
                matmul_tiled(a, b_t, out, M, K, N);
            }
#endif
        } else {
            if (use_avx2) {
                matmul_avx2(a, b_t, out, M, K, N);
            } else {
                matmul_tiled(a, b_t, out, M, K, N);
            }
        }
    }

}  // namespace helix
