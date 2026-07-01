#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "core/tensor_impl.hpp"

namespace helix {

    /**
     * @class Tensor
     * @brief The core multi-dimensional array class in HELIX.
     *
     * Tensor is the fundamental data structure used for all mathematical operations
     * in the framework. It acts as a smart pointer over a `TensorImpl` which holds
     * the actual data, shape, stride, and autograd metadata.
     */
    class Tensor {
    public:
        /**
         * @brief Creates an uninitialized Tensor with the given shape.
         * @param shape The dimensions of the new Tensor.
         * @return A new Tensor instance.
         */
        static Tensor empty(const Shape& shape);

        /**
         * @brief Creates a Tensor filled with zeros.
         * @param shape The dimensions of the new Tensor.
         * @return A new Tensor instance.
         */
        static Tensor zeros(const Shape& shape);

        /**
         * @brief Creates a Tensor filled with ones.
         * @param shape The dimensions of the new Tensor.
         * @return A new Tensor instance.
         */
        static Tensor ones(const Shape& shape);

        /**
         * @brief Creates a Tensor filled with a specific value.
         * @param shape The dimensions of the new Tensor.
         * @param value The value to fill the Tensor with.
         * @return A new Tensor instance.
         */
        static Tensor full(const Shape& shape, float value);

        /**
         * @brief Creates a Tensor with values drawn from a standard normal distribution.
         * @param shape The dimensions of the new Tensor.
         * @return A new Tensor instance.
         */
        static Tensor randn(const Shape& shape);

        /**
         * @brief Constructs an empty (null) Tensor.
         */
        Tensor();

        /**
         * @brief Constructs a new Tensor with the specified shape, data type, and device.
         * @param shape The dimensions of the Tensor.
         * @param dtype The data type (default: Float32).
         * @param device The device to allocate memory on (default: CPU).
         */
        explicit Tensor(Shape shape, DType dtype = DType::Float32, Device device = Device(DeviceType::CPU));

        /**
         * @brief Constructs a new Tensor and copies data from a std::vector.
         * @param data The 1D vector containing the data.
         * @param shape The shape of the new Tensor.
         */
        Tensor(const std::vector<float>& data, Shape shape);

        /**
         * @brief Gets the underlying implementation.
         * @return A shared pointer to the TensorImpl.
         */
        const std::shared_ptr<TensorImpl>& impl() const { return impl_; }

        const Shape& shape() const;
        const Stride& stride() const;
        DType dtype() const;
        Device device() const;
        size_t numel() const;
        size_t rank() const;

        float* data_ptr();
        const float* data_ptr() const;

        float item() const;
        float item(const std::vector<size_t>& indices) const;
        void set_item(const std::vector<size_t>& indices, float value);

        Tensor(const Tensor&) = default;
        Tensor& operator=(const Tensor&) = default;
        Tensor(Tensor&&) noexcept = default;
        Tensor& operator=(Tensor&&) noexcept = default;

        /**
         * @brief Returns a new Tensor with the same data but a different shape.
         * @param new_shape The target shape.
         * @return A viewed Tensor. Throws if contiguous layout is violated.
         */
        Tensor view(Shape new_shape) const;
        Tensor reshape(Shape new_shape) const;
        Tensor transpose(size_t dim0, size_t dim1) const;
        Tensor flatten() const;
        Tensor slice(size_t dim, size_t start, size_t end) const;

        /**
         * @brief Broadcasts the Tensor to a new shape using stride manipulation.
         * @param new_shape The target shape to broadcast to.
         * @return A broadcasted Tensor view.
         */
        Tensor broadcast_to(Shape new_shape) const;

        Tensor operator+(const Tensor& other) const;
        Tensor operator-(const Tensor& other) const;
        Tensor operator*(const Tensor& other) const;
        Tensor operator/(const Tensor& other) const;
        Tensor& add_(const Tensor& other);

        Tensor operator+(float scalar) const;
        Tensor operator-(float scalar) const;
        Tensor operator*(float scalar) const;
        Tensor operator/(float scalar) const;

        /**
         * @brief Performs matrix multiplication between this Tensor and another.
         * @param other The right-hand side Tensor.
         * @return The resulting Tensor.
         */
        Tensor matmul(const Tensor& other) const;

        Tensor operator-() const;
        Tensor exp() const;
        Tensor log() const;
        Tensor sqrt() const;
        Tensor relu() const;
        Tensor pow(float exponent) const;

        /**
         * @brief Checks if the Tensor requires gradient computation.
         * @return True if gradients are tracked, false otherwise.
         */
        bool requires_grad() const;
        void set_requires_grad(bool req) const;
        bool has_grad() const;
        Tensor& grad();
        const Tensor& grad() const;

        /**
         * @brief Computes the gradient of current tensor w.r.t. graph leaves.
         * @param grad_outputs Optional starting gradient for non-scalar outputs.
         */
        void backward(const std::vector<Tensor>& grad_outputs = {});
        Tensor detach() const;

        Tensor sum(std::optional<size_t> axis = std::nullopt, bool keepdim = false) const;
        Tensor mean(std::optional<size_t> axis = std::nullopt, bool keepdim = false) const;

        Tensor clone() const;
        void copy_(const Tensor& src);
        void zero_();
        Tensor contiguous() const;
        bool is_contiguous() const;
        bool is_shared() const;

    private:
        explicit Tensor(std::shared_ptr<TensorImpl> impl);
        std::shared_ptr<TensorImpl> impl_;
    };

}  // namespace helix
