#include <algorithm>
#include <cmath>
#include <cstddef> // for std::ptrdiff_t

#include "backend/cpu_backend.hpp"

namespace helix {

    void CPUBackend::cross_entropy(
        const float* pred, const float* target, float* loss_out, float* log_softmax_out, size_t N, size_t C
    ) {
        double total_loss = 0.0;
        const std::ptrdiff_t N_signed = static_cast<std::ptrdiff_t>(N);

#pragma omp parallel for reduction(+ : total_loss)
        for (std::ptrdiff_t i = 0; i < N_signed; ++i) {
            const float* p_row = pred + i * C;
            const float* t_row = target + i * C;
            float* ls_row = log_softmax_out + i * C;

            // 1. Find max for numerical stability (Log-Sum-Exp Trick)
            float max_val = p_row[0];
            for (size_t j = 1; j < C; ++j) {
                max_val = std::max(max_val, p_row[j]);
            }

            // 2. Compute sum of exp(shifted)
            double sum_exp = 0.0;
            for (size_t j = 0; j < C; ++j) {
                sum_exp += std::exp(p_row[j] - max_val);
            }

            // 3. Compute log_softmax and accumulate loss
            double log_sum_exp = std::log(sum_exp);
            double row_loss = 0.0;

            for (size_t j = 0; j < C; ++j) {
                float shifted = p_row[j] - max_val;
                float log_softmax = shifted - static_cast<float>(log_sum_exp);
                ls_row[j] = log_softmax;

                // Accumulate loss: -target * log_softmax
                if (t_row[j] > 0.0f) {
                    row_loss -= t_row[j] * log_softmax;
                }
            }

            total_loss += row_loss;
        }

        // 4. Mean over batch size N
        loss_out[0] = static_cast<float>(total_loss / N);
    }

}  // namespace helix
