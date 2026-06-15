#pragma once

#include "core/shape.hpp"
#include "core/stride.hpp"

namespace helix {

    // Computes the broadcasted shape of two shapes following Numpy broadcasting rules.
    // Throws std::invalid_argument if the shapes are not broadcastable.
    Shape compute_broadcast_shape(const Shape& a, const Shape& b);

    // Computes the new strides for a tensor when broadcasted to a target shape.
    // The resulting stride will have 0 for dimensions that were broadcasted.
    Stride compute_broadcast_strides(
        const Shape& original_shape, const Stride& original_stride, const Shape& target_shape
    );

}  // namespace helix
