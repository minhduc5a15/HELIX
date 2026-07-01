#pragma once

#include "optim/optimizer.hpp"

namespace helix {

    /**
     * @class SGD
     * @brief Implements stochastic gradient descent (optionally with momentum).
     */
    class SGD : public Optimizer {
    public:
        /**
         * @brief Constructs an SGD optimizer.
         * @param params Vector of parameter Tensors to optimize.
         * @param lr Learning rate.
         */
        SGD(std::vector<Tensor> params, float lr);

        /**
         * @brief Performs a single optimization step.
         */
        void step() override;

        /**
         * @brief Gets the current learning rate.
         * @return The learning rate value.
         */
        float learning_rate() const { return learning_rate_; }

        /**
         * @brief Updates the learning rate.
         * @param lr The new learning rate value.
         */
        void set_learning_rate(const float lr) { learning_rate_ = lr; }

    private:
        float learning_rate_;
    };

}  // namespace helix
