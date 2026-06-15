#pragma once

#include <memory>
#include <vector>

#include "core/tensor_impl.hpp"

namespace helix {

    class Tensor {
    public:
        // Empty tensor (e.g. for default initialization before assignment)
        Tensor();

        // Create a new tensor with given shape
        explicit Tensor(Shape shape, DType dtype = DType::Float32, Device device = Device(DeviceType::CPU));

        // Create tensor from 1D data (copies data to storage)
        Tensor(const std::vector<float>& data, Shape shape);

        // Access underlying impl
        const std::shared_ptr<TensorImpl>& impl() const { return impl_; }

        // Properties
        const Shape& shape() const;
        const Stride& stride() const;
        DType dtype() const;
        Device device() const;
        size_t numel() const;
        size_t rank() const;

        // Direct memory access (assuming Float32 for now)
        float* data_ptr();
        const float* data_ptr() const;

        // Value access via indices
        float item() const;  // For scalar or 1-element tensor
        float item(const std::vector<size_t>& indices) const;
        void set_item(const std::vector<size_t>& indices, float value);

        // Copy and Move Semantics (Shallow copy / reference counting)
        Tensor(const Tensor&) = default;
        Tensor& operator=(const Tensor&) = default;
        Tensor(Tensor&&) noexcept = default;
        Tensor& operator=(Tensor&&) noexcept = default;

        // View and Shape operations
        Tensor view(Shape new_shape) const;
        Tensor reshape(Shape new_shape) const;
        Tensor transpose(size_t dim0, size_t dim1) const;
        // Broadcast
        Tensor broadcast_to(Shape new_shape) const;

        // Mathematical Operations (Element-wise)
        Tensor operator+(const Tensor& other) const;
        Tensor operator-(const Tensor& other) const;
        Tensor operator*(const Tensor& other) const;
        Tensor operator/(const Tensor& other) const;
        Tensor matmul(const Tensor& other) const;

        // Unary
        Tensor operator-() const;

        // Memory operations
        Tensor clone() const;
        Tensor contiguous() const;
        bool is_contiguous() const;

    private:
        // Internal constructor used for views and other internals
        explicit Tensor(std::shared_ptr<TensorImpl> impl);

        std::shared_ptr<TensorImpl> impl_;
    };

}  // namespace helix
