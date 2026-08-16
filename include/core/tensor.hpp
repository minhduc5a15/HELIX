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
        static auto empty(const Shape& shape) -> Tensor;

        /**
         * @brief Creates a Tensor filled with zeros.
         * @param shape The dimensions of the new Tensor.
         * @return A new Tensor instance.
         */
        static auto zeros(const Shape& shape) -> Tensor;

        /**
         * @brief Creates a Tensor filled with ones.
         * @param shape The dimensions of the new Tensor.
         * @return A new Tensor instance.
         */
        static auto ones(const Shape& shape) -> Tensor;

        /**
         * @brief Creates a Tensor filled with a specific value.
         * @param shape The dimensions of the new Tensor.
         * @param value The value to fill the Tensor with.
         * @return A new Tensor instance.
         */
        static auto full(const Shape& shape, float value) -> Tensor;

        /**
         * @brief Creates a Tensor with values drawn from a standard normal distribution.
         * @param shape The dimensions of the new Tensor.
         * @return A new Tensor instance.
         */
        static auto randn(const Shape& shape) -> Tensor;

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
        [[nodiscard]] auto impl() const -> const std::shared_ptr<TensorImpl>& { return impl_; }

        [[nodiscard]] auto shape() const -> const Shape&;
        [[nodiscard]] auto stride() const -> const Stride&;
        [[nodiscard]] auto dtype() const -> DType;
        [[nodiscard]] auto device() const -> Device;
        [[nodiscard]] auto numel() const -> size_t;
        [[nodiscard]] auto rank() const -> size_t;

        auto data_ptr() -> float*;
        [[nodiscard]] auto data_ptr() const -> const float*;

        /**
         * @brief Gets the current version of the tensor (used for autograd inplace checks).
         * @return The version number.
         */
        [[nodiscard]] auto version() const -> uint32_t;

        /**
         * @brief Increments the tensor version, tracking in-place modifications.
         */
        void increment_version();

        /**
         * @brief Extracts a single float value if the tensor is a scalar.
         * @return The float value.
         */
        [[nodiscard]] auto item() const -> float;

        /**
         * @brief Gets the float value at the specified indices.
         * @param indices The multi-dimensional coordinates.
         * @return The value at the coordinate.
         */
        [[nodiscard]] auto item(const std::vector<size_t>& indices) const -> float;

        /**
         * @brief Sets a float value at the specified indices (in-place operation).
         * @param indices The multi-dimensional coordinates.
         * @param value The new value to set.
         */
        void set_item(const std::vector<size_t>& indices, float value);

        Tensor(const Tensor&) = default;
        auto operator=(const Tensor&) -> Tensor& = default;
        Tensor(Tensor&&) noexcept = default;
        auto operator=(Tensor&&) noexcept -> Tensor& = default;

        /**
         * @brief Returns a new Tensor with the same data but a different shape.
         * @param new_shape The target shape.
         * @return A viewed Tensor. Throws if contiguous layout is violated.
         */
        [[nodiscard]] auto view(Shape new_shape) const -> Tensor;
        /**
         * @brief Reshapes the tensor in-place if contiguous, or copies if not.
         * @param new_shape The desired target shape.
         * @return A reshaped tensor.
         */
        [[nodiscard]] auto reshape(Shape new_shape) const -> Tensor;

        /**
         * @brief Transposes two dimensions of the tensor.
         * @param dim0 The first dimension to transpose.
         * @param dim1 The second dimension to transpose.
         * @return A transposed tensor view.
         */
        [[nodiscard]] auto transpose(size_t dim0, size_t dim1) const -> Tensor;

        /**
         * @brief Flattens the tensor into a 1D tensor.
         * @return A flattened tensor.
         */
        [[nodiscard]] auto flatten() const -> Tensor;

        /**
         * @brief Slices the tensor along a specific dimension.
         * @param dim The dimension to slice.
         * @param start The starting index.
         * @param end The ending index.
         * @return A tensor view representing the slice.
         */
        [[nodiscard]] auto slice(size_t dim, size_t start, size_t end) const -> Tensor;

        /**
         * @brief Broadcasts the Tensor to a new shape using stride manipulation.
         * @param new_shape The target shape to broadcast to.
         * @return A broadcasted Tensor view.
         */
        [[nodiscard]] auto broadcast_to(Shape new_shape) const -> Tensor;

        auto operator+(const Tensor& other) const -> Tensor;
        auto operator-(const Tensor& other) const -> Tensor;
        auto operator*(const Tensor& other) const -> Tensor;
        auto operator/(const Tensor& other) const -> Tensor;
        auto add_(const Tensor& other) -> Tensor&;

        auto operator+(float scalar) const -> Tensor;
        auto operator-(float scalar) const -> Tensor;
        auto operator*(float scalar) const -> Tensor;
        auto operator/(float scalar) const -> Tensor;

        /**
         * @brief Performs matrix multiplication between this Tensor and another.
         * @param other The right-hand side Tensor.
         * @return The resulting Tensor.
         */
        [[nodiscard]] auto matmul(const Tensor& other) const -> Tensor;

        /**
         * @brief Element-wise negation.
         * @return The negated tensor.
         */
        auto operator-() const -> Tensor;

        /**
         * @brief Element-wise exponential (e^x).
         * @return The resulting tensor.
         */
        [[nodiscard]] auto exp() const -> Tensor;
        [[nodiscard]] auto tanh() const -> Tensor;

        /**
         * @brief Element-wise natural logarithm (ln(x)).
         * @return The resulting tensor.
         */
        [[nodiscard]] auto log() const -> Tensor;

        /**
         * @brief Element-wise square root.
         * @return The resulting tensor.
         */
        [[nodiscard]] auto sqrt() const -> Tensor;

        /**
         * @brief Rectified Linear Unit (ReLU) activation.
         * @return The resulting tensor.
         */
        [[nodiscard]] auto relu() const -> Tensor;

        /**
         * @brief Element-wise power.
         * @param exponent The power to raise elements to.
         * @return The resulting tensor.
         */
        [[nodiscard]] auto pow(float exponent) const -> Tensor;

        /**
         * @brief Checks if the Tensor requires gradient computation.
         * @return True if gradients are tracked, false otherwise.
         */
        [[nodiscard]] auto requires_grad() const -> bool;
        void set_requires_grad(bool req) const;
        [[nodiscard]] auto has_grad() const -> bool;
        auto grad() -> Tensor&;
        [[nodiscard]] auto grad() const -> const Tensor&;

        /**
         * @brief Computes the gradient of current tensor w.r.t. graph leaves.
         * @param grad_outputs Optional starting gradient for non-scalar outputs.
         * @param retain_graph If true, preserves the compute graph for subsequent backward passes.
         */
        void backward(const std::vector<Tensor>& grad_outputs = {}, bool retain_graph = false);

        /**
         * @brief Detaches the tensor from the autograd graph.
         * @return A new tensor sharing the same memory, but requiring no gradients.
         */
        [[nodiscard]] auto detach() const -> Tensor;

        [[nodiscard]] auto sum(std::optional<size_t> axis = std::nullopt, bool keepdim = false) const -> Tensor;
        [[nodiscard]] auto mean(std::optional<size_t> axis = std::nullopt, bool keepdim = false) const -> Tensor;

        /**
         * @brief Creates a deep copy of the tensor, including its memory.
         * @return A cloned tensor.
         */
        [[nodiscard]] auto clone() const -> Tensor;

        /**
         * @brief Copies data from another tensor into this one (in-place).
         * @param src The source tensor.
         */
        void copy_(const Tensor& src);

        /**
         * @brief Fills the tensor with zeros (in-place).
         */
        void zero_();

        /**
         * @brief Ensures the tensor memory is contiguous.
         * @return A contiguous tensor (may be a copy or a reference if already contiguous).
         */
        [[nodiscard]] auto contiguous() const -> Tensor;

        /**
         * @brief Checks if the memory layout is contiguous (no strides).
         * @return True if contiguous.
         */
        [[nodiscard]] auto is_contiguous() const -> bool;

        /**
         * @brief Checks if this tensor shares memory with another tensor (is a view).
         * @return True if it is a view.
         */
        [[nodiscard]] auto is_shared() const -> bool;

        /**
         * @brief Checks if the tensor has internal memory overlap (e.g., due to broadcasting).
         * @return True if multiple logical indices map to the same physical memory location.
         */
        [[nodiscard]] auto has_internal_overlap() const -> bool;

    public:
        explicit Tensor(std::shared_ptr<TensorImpl> impl);

    private:
        std::shared_ptr<TensorImpl> impl_;
    };

}  // namespace helix
