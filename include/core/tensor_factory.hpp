#pragma once

#include "core/shape.hpp"
#include "core/tensor.hpp"

namespace helix {

    class TensorFactory {
    public:
        static Tensor empty(const Shape& shape);
        static Tensor zeros(const Shape& shape);
        static Tensor ones(const Shape& shape);
        static Tensor full(const Shape& shape, float value);
        static Tensor randn(const Shape& shape);
    };

}  // namespace helix
