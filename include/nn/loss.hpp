#pragma once

#include "core/tensor.hpp"

namespace helix {

    Tensor mse_loss(const Tensor& pred, const Tensor& target);

}  // namespace helix
