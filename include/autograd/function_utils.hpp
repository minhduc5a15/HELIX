#pragma once

#include "core/shape.hpp"
#include "core/tensor.hpp"

namespace helix {

    // Helper to reduce the gradient back to the original target shape if it was broadcasted
    Tensor sum_to_shape(const Tensor& grad, const Shape& target_shape);

}  // namespace helix
