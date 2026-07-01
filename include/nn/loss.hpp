#pragma once

#include "core/tensor.hpp"

namespace helix {

    /**
     * @brief Creates a criterion that measures the mean squared error (squared L2 norm) between
     * each element in the input x and target y.
     *
     * @param pred The predicted Tensor.
     * @param target The ground truth Tensor.
     * @return A scalar Tensor representing the MSE Loss.
     */
    Tensor mse_loss(const Tensor& pred, const Tensor& target);

}  // namespace helix
