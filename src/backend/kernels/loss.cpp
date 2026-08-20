#include <algorithm>
#include <cmath>
#include <cstddef>  // for std::ptrdiff_t

#include "backend/cpu_backend.hpp"

namespace helix {

    template <typename T>
    void CPUBackend::cross_entropy(
        const T* pred, const T* target, T* loss_out, T* log_softmax_out, const size_t N, const size_t C
    ) {
        double total_loss = 0.0;
        const std::ptrdiff_t N_signed = static_cast<std::ptrdiff_t>(N);

#pragma omp parallel for reduction(+ : total_loss)
        for (std::ptrdiff_t i = 0; i < N_signed; ++i) {
            const T* p_row = pred + i * C;
            const T* t_row = target + i * C;
            T* ls_row = log_softmax_out + i * C;

            // 1. Find max for numerical stability (Log-Sum-Exp Trick)
            T max_val = p_row[0];
            for (size_t j = 1; j < C; ++j) {
                max_val = std::max(max_val, p_row[j]);
            }

            // 2. Compute sum of exp(shifted)
            double sum_exp = 0.0;
            for (size_t j = 0; j < C; ++j) {
                sum_exp += std::exp(static_cast<double>(p_row[j] - max_val));
            }

            // 3. Compute log_softmax and accumulate loss
            const double log_sum_exp = std::log(sum_exp);
            double row_loss = 0.0;

            for (size_t j = 0; j < C; ++j) {
                const double shifted = static_cast<double>(p_row[j] - max_val);
                const double log_softmax = shifted - log_sum_exp;
                ls_row[j] = static_cast<T>(log_softmax);

                // Accumulate loss: -target * log_softmax
                if (t_row[j] > static_cast<T>(0)) {
                    row_loss -= static_cast<double>(t_row[j]) * log_softmax;
                }
            }

            total_loss += row_loss;
        }

        // 4. Mean over batch size N
        loss_out[0] = static_cast<T>(total_loss / static_cast<double>(N));
    }

    template void CPUBackend::cross_entropy<float>(const float*, const float*, float*, float*, size_t, size_t);
    template void CPUBackend::cross_entropy<double>(const double*, const double*, double*, double*, size_t, size_t);
    template void CPUBackend::cross_entropy<int32_t>(
        const int32_t*, const int32_t*, int32_t*, int32_t*, size_t, size_t
    );
    template void CPUBackend::cross_entropy<int64_t>(
        const int64_t*, const int64_t*, int64_t*, int64_t*, size_t, size_t
    );

}  // namespace helix
