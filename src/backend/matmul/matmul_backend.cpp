#include <algorithm>

#include "backend/cpu_backend.hpp"
#include "core/autotuner.hpp"
#include "core/cpu_utils.hpp"
#include "matmul_kernel.hpp"

namespace helix {
    // Helper to check AVX2 support at runtime
    inline bool supports_avx2_backend() {
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

    template <typename T>
    void generic_blocked_matmul(const T* a, const T* b, T* out, const size_t M, const size_t K, const size_t N) {
        for (size_t i = 0; i < M * N; ++i) {
            out[i] = static_cast<T>(0);
        }

        constexpr size_t BLOCK = 64;  // Block size for generic matmul

#if defined(_OPENMP)
#pragma omp parallel for collapse(2)
#endif
        for (size_t ih = 0; ih < M; ih += BLOCK) {
            for (size_t jh = 0; jh < N; jh += BLOCK) {
                const size_t i_end = std::min(ih + BLOCK, M);
                const size_t j_end = std::min(jh + BLOCK, N);
                for (size_t kh = 0; kh < K; kh += BLOCK) {
                    const size_t k_end = std::min(kh + BLOCK, K);

                    // Loop order i, k, j ensures contiguous memory access for out and b
                    for (size_t i = ih; i < i_end; ++i) {
                        for (size_t k = kh; k < k_end; ++k) {
                            const T a_val = a[i * K + k];
                            for (size_t j = jh; j < j_end; ++j) {
                                out[i * N + j] += a_val * b[k * N + j];
                            }
                        }
                    }
                }
            }
        }
    }

    template <typename T>
    void CPUBackend::matmul(
        const T* a, const T* b, T* out, const size_t M, const size_t K, const size_t N, MatMulStrategy strategy
    ) {
        if constexpr (std::is_same_v<T, float>) {
            if (strategy == MatMulStrategy::Auto) {
#if defined(_OPENMP)
                const size_t volume = static_cast<size_t>(M) * static_cast<size_t>(N) * static_cast<size_t>(K);
                const size_t threshold = AutoTuner::get_instance().get_omp_threshold();
                if (volume >= threshold) {
                    strategy = MatMulStrategy::OpenMP;
                } else {
                    strategy = supports_avx2_backend() ? MatMulStrategy::AVX2 : MatMulStrategy::Blocked;
                }
#else
                strategy = supports_avx2_backend() ? MatMulStrategy::AVX2 : MatMulStrategy::Blocked;
#endif
            }

            switch (strategy) {
                case MatMulStrategy::Naive:
                    naive_matmul(a, b, out, M, K, N);
                    break;
                case MatMulStrategy::Blocked:
                    blocked_matmul(a, b, out, M, K, N);
                    break;
                case MatMulStrategy::AVX2:
                    avx2_micro_matmul(a, b, out, M, K, N);
                    break;
                case MatMulStrategy::OpenMP:
                    openmp_matmul(a, b, out, M, K, N);
                    break;
                default:
                    blocked_matmul(a, b, out, M, K, N);
                    break;
            }
        } else {
            // For non-float types (e.g. integers), AVX2 math is not supported. Use fallback OpenMP matmul.
            generic_blocked_matmul(a, b, out, M, K, N);
        }
    }

    template void CPUBackend::matmul<float>(const float*, const float*, float*, size_t, size_t, size_t, MatMulStrategy);
    template void CPUBackend::matmul<double>(
        const double*, const double*, double*, size_t, size_t, size_t, MatMulStrategy
    );
    template void CPUBackend::matmul<int32_t>(
        const int32_t*, const int32_t*, int32_t*, size_t, size_t, size_t, MatMulStrategy
    );
    template void CPUBackend::matmul<int64_t>(
        const int64_t*, const int64_t*, int64_t*, size_t, size_t, size_t, MatMulStrategy
    );

}  // namespace helix
