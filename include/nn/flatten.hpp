#pragma once

#include <cstdint>

#include "core/tensor.hpp"
#include "nn/module.hpp"

namespace helix {

    /**
     * @class Flatten
     * @brief A neural network module that flattens a consecutive range of dimensions into a single dimension.
     *
     * Defaults to start_dim = 1, end_dim = -1, preserving the batch dimension (dim 0).
     */
    class Flatten : public Module {
    public:
        /**
         * @brief Constructs a Flatten module.
         * @param start_dim First dimension to flatten (default: 1).
         * @param end_dim Last dimension to flatten (inclusive, default: -1).
         */
        explicit Flatten(int64_t start_dim = 1, int64_t end_dim = -1);

        /**
         * @brief Forwards the input through the flattening operation.
         * @param input The input Tensor.
         * @return The flattened Tensor.
         */
        Tensor forward(const Tensor& input) override;

        [[nodiscard]] int64_t start_dim() const noexcept { return start_dim_; }
        [[nodiscard]] int64_t end_dim() const noexcept { return end_dim_; }

    private:
        int64_t start_dim_{1};
        int64_t end_dim_{-1};
    };

}  // namespace helix
