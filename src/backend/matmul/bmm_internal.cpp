#include "bmm_internal.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "core/cpu_utils.hpp"
#include "core/shape.hpp"
#include "matmul_kernel.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace helix::internal {

    template <typename T>
    void bmm_serial_batches(
        const T* a,
        const T* b,
        T* out,
        const size_t B,
        const size_t M,
        const size_t K,
        const size_t N,
        MatMulStrategy strategy
    ) {
        if (B == 0 || M == 0 || N == 0) {
            return;
        }
        if (K == 0) {
            std::fill_n(out, Shape{B, M, N}.numel(), static_cast<T>(0));
            return;
        }

        const size_t a_stride = Shape{M, K}.numel();
        const size_t b_stride = Shape{K, N}.numel();
        const size_t out_stride = Shape{M, N}.numel();

        for (size_t batch = 0; batch < B; ++batch) {
            const T* a_batch = a + batch * a_stride;
            const T* b_batch = b + batch * b_stride;
            T* out_batch = out + batch * out_stride;
            CPUBackend::matmul(a_batch, b_batch, out_batch, M, K, N, strategy);
        }
    }

    template void bmm_serial_batches<float>(
        const float*, const float*, float*, size_t, size_t, size_t, size_t, MatMulStrategy
    );
    template void bmm_serial_batches<double>(
        const double*, const double*, double*, size_t, size_t, size_t, size_t, MatMulStrategy
    );
    template void bmm_serial_batches<int32_t>(
        const int32_t*, const int32_t*, int32_t*, size_t, size_t, size_t, size_t, MatMulStrategy
    );
    template void bmm_serial_batches<int64_t>(
        const int64_t*, const int64_t*, int64_t*, size_t, size_t, size_t, size_t, MatMulStrategy
    );

    void bmm_parallel_batches_float(
        const float* a, const float* b, float* out, const size_t B, const size_t M, const size_t K, const size_t N
    ) {
        if (B == 0 || M == 0 || N == 0) {
            return;
        }
        if (K == 0) {
            std::fill_n(out, Shape{B, M, N}.numel(), 0.0f);
            return;
        }

        const size_t a_stride = Shape{M, K}.numel();
        const size_t b_stride = Shape{K, N}.numel();
        const size_t out_stride = Shape{M, N}.numel();
        const bool use_avx2 = supports_avx2_backend();
        const ptrdiff_t B_s = static_cast<ptrdiff_t>(B);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (ptrdiff_t batch = 0; batch < B_s; ++batch) {
            const float* a_batch = a + batch * a_stride;
            const float* b_batch = b + batch * b_stride;
            float* out_batch = out + batch * out_stride;

            if (use_avx2) {
                avx2_micro_matmul(a_batch, b_batch, out_batch, M, K, N);
            } else {
                blocked_matmul(a_batch, b_batch, out_batch, M, K, N);
            }
        }
    }

}  // namespace helix::internal
