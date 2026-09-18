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

    /**
     * @brief Concatenates a sequence of tensors along a given dimension.
     * @param tensors Non-empty list of tensors with matching shapes except along `dim`.
     * @param dim Dimension along which to concatenate.
     * @return A new contiguous tensor containing the concatenated data.
     */
    Tensor cat(const std::vector<Tensor>& tensors, size_t dim = 0);

}  // namespace helix
