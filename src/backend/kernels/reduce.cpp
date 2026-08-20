#include <type_traits>

#include "backend/cpu_backend.hpp"
#include "backend/simd_backend.hpp"

namespace helix {

    template <typename T>
    void CPUBackend::sum(
        const T* input, T* output, const size_t outer_size, const size_t dim_size, const size_t inner_size
    ) {
        if constexpr (std::is_same_v<T, float>) {
            if (SIMDBackend::is_supported()) {
                SIMDBackend::sum(input, output, outer_size, dim_size, inner_size);
                return;
            }
        }
        for (size_t i = 0; i < outer_size; ++i) {
            for (size_t k = 0; k < inner_size; ++k) {
                T sum_val = 0;
                for (size_t j = 0; j < dim_size; ++j) {
                    sum_val += input[i * (dim_size * inner_size) + j * inner_size + k];
                }
                output[i * inner_size + k] = sum_val;
            }
        }
    }

    template <typename T>
    void CPUBackend::mean(
        const T* input, T* output, const size_t outer_size, const size_t dim_size, const size_t inner_size
    ) {
        if constexpr (std::is_same_v<T, float>) {
            if (SIMDBackend::is_supported()) {
                SIMDBackend::mean(input, output, outer_size, dim_size, inner_size);
                return;
            }
        }
        const double scale = 1.0 / static_cast<double>(dim_size);
        for (size_t i = 0; i < outer_size; ++i) {
            for (size_t k = 0; k < inner_size; ++k) {
                double sum_val = 0.0;
                for (size_t j = 0; j < dim_size; ++j) {
                    sum_val += static_cast<double>(input[i * (dim_size * inner_size) + j * inner_size + k]);
                }
                output[i * inner_size + k] = static_cast<T>(sum_val * scale);
            }
        }
    }

    template void CPUBackend::sum<float>(const float*, float*, size_t, size_t, size_t);
    template void CPUBackend::mean<float>(const float*, float*, size_t, size_t, size_t);
    template void CPUBackend::sum<double>(const double*, double*, size_t, size_t, size_t);
    template void CPUBackend::mean<double>(const double*, double*, size_t, size_t, size_t);
    template void CPUBackend::sum<int32_t>(const int32_t*, int32_t*, size_t, size_t, size_t);
    template void CPUBackend::mean<int32_t>(const int32_t*, int32_t*, size_t, size_t, size_t);
    template void CPUBackend::sum<int64_t>(const int64_t*, int64_t*, size_t, size_t, size_t);
    template void CPUBackend::mean<int64_t>(const int64_t*, int64_t*, size_t, size_t, size_t);

}  // namespace helix
