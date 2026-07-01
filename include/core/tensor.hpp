#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "core/tensor_impl.hpp"

namespace helix {

    class Tensor {
    public:
        // Factory Methods
        static Tensor empty(const Shape& shape);
        static Tensor zeros(const Shape& shape);
        static Tensor ones(const Shape& shape);
        static Tensor full(const Shape& shape, float value);
        static Tensor randn(const Shape& shape);

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
        Tensor flatten() const;
        Tensor slice(size_t dim, size_t start, size_t end) const;

        // Broadcast
        Tensor broadcast_to(Shape new_shape) const;

        // Mathematical Operations (Element-wise)
        Tensor operator+(const Tensor& other) const;
        Tensor operator-(const Tensor& other) const;
        Tensor operator*(const Tensor& other) const;
        Tensor operator/(const Tensor& other) const;
        Tensor& add_(const Tensor& other);

        // Scalar arithmetic
        Tensor operator+(float scalar) const;
        Tensor operator-(float scalar) const;
        Tensor operator*(float scalar) const;
        Tensor operator/(float scalar) const;
        Tensor matmul(const Tensor& other) const;

        // Unary
        Tensor operator-() const;
        Tensor exp() const;
        Tensor log() const;
        Tensor sqrt() const;
        Tensor relu() const;
        Tensor pow(float exponent) const;

        // Autograd API
        bool requires_grad() const;
        void set_requires_grad(bool req) const;
        bool has_grad() const;
        Tensor& grad();
        const Tensor& grad() const;
        void backward(const std::vector<Tensor>& grad_outputs = {});
        Tensor detach() const;

        // Reduce Operations
        Tensor sum(std::optional<size_t> axis = std::nullopt, bool keepdim = false) const;
        Tensor mean(std::optional<size_t> axis = std::nullopt, bool keepdim = false) const;

        // Memory operations
        Tensor clone() const;
        void copy_(const Tensor& src);
        void zero_();
        Tensor contiguous() const;
        bool is_contiguous() const;
        bool is_shared() const;

    private:
        // Internal constructor used for views and other internals
        explicit Tensor(std::shared_ptr<TensorImpl> impl);

        std::shared_ptr<TensorImpl> impl_;
    };

}  // namespace helix
