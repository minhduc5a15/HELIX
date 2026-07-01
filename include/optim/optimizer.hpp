#pragma once

#include <vector>

#include "core/tensor.hpp"

namespace helix {

    /**
     * @class Optimizer
     * @brief Base class for all optimizers.
     */
    class Optimizer {
    protected:
        std::vector<Tensor> params_;

    public:
        /**
         * @brief Constructs a new Optimizer.
         * @param params An iterable of Tensors to optimize.
         */
        explicit Optimizer(std::vector<Tensor> params) : params_(std::move(params)) {}
        virtual ~Optimizer() = default;

        /**
         * @brief Performs a single optimization step (parameter update).
         */
        virtual void step() = 0;

        /**
         * @brief Clears the gradients of all optimized Tensors.
         *
         * This should be called before the backward pass of a new training iteration.
         */
        void zero_grad() {
            for (auto& param : params_) {
                if (param.requires_grad() && param.has_grad()) {
                    param.grad().zero_();
                }
            }
        }

        /**
         * @brief Returns the parameters managed by this optimizer.
         * @return A const reference to the vector of parameter Tensors.
         */
        const std::vector<Tensor>& parameters() const { return params_; }
    };

}  // namespace helix
