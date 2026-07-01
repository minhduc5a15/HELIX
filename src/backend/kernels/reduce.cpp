#include "backend/cpu_backend.hpp"
#include "backend/simd_backend.hpp"

namespace helix {

    void CPUBackend::sum(
        const float* input, float* output, const size_t outer_size, const size_t dim_size, const size_t inner_size
    ) {
        if (SIMDBackend::is_supported()) {
            SIMDBackend::sum(input, output, outer_size, dim_size, inner_size);
        } else {
            for (size_t i = 0; i < outer_size; ++i) {
                for (size_t k = 0; k < inner_size; ++k) {
                    float sum_val = 0.0f;
                    for (size_t j = 0; j < dim_size; ++j) {
                        sum_val += input[i * (dim_size * inner_size) + j * inner_size + k];
                    }
                    output[i * inner_size + k] = sum_val;
                }
            }
        }
    }

    void CPUBackend::mean(
        const float* input, float* output, const size_t outer_size, const size_t dim_size, const size_t inner_size
    ) {
        if (SIMDBackend::is_supported()) {
            SIMDBackend::mean(input, output, outer_size, dim_size, inner_size);
        } else {
            const float scale = 1.0f / static_cast<float>(dim_size);
            for (size_t i = 0; i < outer_size; ++i) {
                for (size_t k = 0; k < inner_size; ++k) {
                    float sum_val = 0.0f;
                    for (size_t j = 0; j < dim_size; ++j) {
                        sum_val += input[i * (dim_size * inner_size) + j * inner_size + k];
                    }
                    output[i * inner_size + k] = sum_val * scale;
                }
            }
        }
    }

}  // namespace helix
