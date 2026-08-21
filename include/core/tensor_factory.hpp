#pragma once

#include "core/shape.hpp"
#include "core/tensor.hpp"

namespace helix {

    class TensorFactory {
    public:
        static Tensor empty(
            const Shape& shape, std::optional<DType> dtype = std::nullopt, std::optional<Device> device = std::nullopt
        );
        static Tensor zeros(
            const Shape& shape, std::optional<DType> dtype = std::nullopt, std::optional<Device> device = std::nullopt
        );
        static Tensor ones(
            const Shape& shape, std::optional<DType> dtype = std::nullopt, std::optional<Device> device = std::nullopt
        );
        static Tensor full(
            const Shape& shape,
            float value,
            std::optional<DType> dtype = std::nullopt,
            std::optional<Device> device = std::nullopt
        );
        static Tensor randn(
            const Shape& shape, std::optional<DType> dtype = std::nullopt, std::optional<Device> device = std::nullopt
        );
    };

}  // namespace helix
