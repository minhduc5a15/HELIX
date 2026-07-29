#include "backend/cpu_backend.hpp"
#include "core/autotuner.hpp"
#include "matmul_kernel.hpp"

namespace helix {
    // Helper to check AVX2 support at runtime
    inline bool supports_avx2_backend() {
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

    void CPUBackend::matmul(
        const float* a, const float* b, float* out, size_t M, size_t K, size_t N, MatMulStrategy strategy
    ) {
        if (strategy == MatMulStrategy::Auto) {
#if defined(_OPENMP)
            size_t volume = (size_t)M * (size_t)N * (size_t)K;
            size_t threshold = AutoTuner::get_instance().get_omp_threshold();
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
    }
}  // namespace helix
