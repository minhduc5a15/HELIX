#include "backend/cpu_backend.hpp"

namespace helix {

    void CPUBackend::sum(const float* input, float* output, size_t outer_size, size_t dim_size, size_t inner_size) {
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

    void CPUBackend::mean(const float* input, float* output, size_t outer_size, size_t dim_size, size_t inner_size) {
        float scale = 1.0f / static_cast<float>(dim_size);
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

}  // namespace helix
