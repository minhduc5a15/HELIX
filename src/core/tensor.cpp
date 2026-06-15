#include "core/tensor.hpp"

#include <cstring>  // for memcpy
#include <stdexcept>

#include "core/broadcast.hpp"
#include "core/dispatcher.hpp"

namespace helix {

    Tensor::Tensor() : impl_(std::make_shared<TensorImpl>(Shape{}, DType::Float32, Device(DeviceType::CPU))) {}

    Tensor::Tensor(Shape shape, DType dtype, Device device)
        : impl_(std::make_shared<TensorImpl>(std::move(shape), dtype, device)) {}

    Tensor::Tensor(const std::vector<float>& data, Shape shape)
        : impl_(std::make_shared<TensorImpl>(std::move(shape), DType::Float32, Device(DeviceType::CPU))) {
        if (data.size() != impl_->shape().numel()) {
            throw std::invalid_argument("Data size does not match tensor shape");
        }

        if (impl_->shape().numel() > 0) {
            std::memcpy(data_ptr(), data.data(), data.size() * sizeof(float));
        }
    }

    Tensor::Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}

    const Shape& Tensor::shape() const { return impl_->shape(); }
    const Stride& Tensor::stride() const { return impl_->stride(); }
    DType Tensor::dtype() const { return impl_->dtype(); }
    Device Tensor::device() const { return impl_->device(); }
    size_t Tensor::numel() const { return impl_->shape().numel(); }
    size_t Tensor::rank() const { return impl_->shape().rank(); }

    float* Tensor::data_ptr() {
        if (dtype() != DType::Float32) throw std::runtime_error("data_ptr() only supports Float32 for now");
        return static_cast<float*>(impl_->data());
    }

    const float* Tensor::data_ptr() const {
        if (dtype() != DType::Float32) throw std::runtime_error("data_ptr() only supports Float32 for now");
        return static_cast<const float*>(impl_->data());
    }

    float Tensor::item() const {
        if (numel() != 1) {
            throw std::runtime_error("item() can only be called on tensors with 1 element");
        }
        return data_ptr()[0];
    }

    float Tensor::item(const std::vector<size_t>& indices) const {
        size_t offset = stride().compute_offset(indices);
        return data_ptr()[offset];
    }

    void Tensor::set_item(const std::vector<size_t>& indices, float value) {
        size_t offset = stride().compute_offset(indices);
        data_ptr()[offset] = value;
    }

    bool Tensor::is_contiguous() const { return impl_->is_contiguous(); }

    Tensor Tensor::view(Shape new_shape) const {
        if (new_shape.numel() != numel()) {
            throw std::invalid_argument("view shape must have the same number of elements");
        }
        if (!is_contiguous()) {
            throw std::runtime_error("view cannot be called on non-contiguous tensor, use reshape instead");
        }
        auto new_impl = std::make_shared<TensorImpl>(
            impl_->storage(),
            impl_->storage_offset(),
            std::move(new_shape),
            Stride::compute_contiguous(new_shape),
            dtype(),
            device()
        );
        return Tensor(new_impl);
    }

    Tensor Tensor::clone() const {
        Tensor new_tensor(shape(), dtype(), device());

        if (is_contiguous()) {
            std::memcpy(new_tensor.data_ptr(), data_ptr(), numel() * sizeof(float));
        } else {
            // N-dimensional iterator for non-contiguous copy
            std::vector<size_t> indices(rank(), 0);
            for (size_t i = 0; i < numel(); ++i) {
                new_tensor.data_ptr()[i] = item(indices);

                for (int j = static_cast<int>(rank()) - 1; j >= 0; --j) {
                    indices[j]++;
                    if (indices[j] < shape()[j]) {
                        break;
                    }
                    indices[j] = 0;
                }
            }
        }
        return new_tensor;
    }

    Tensor Tensor::contiguous() const {
        if (is_contiguous()) return *this;
        return clone();
    }

    Tensor Tensor::reshape(Shape new_shape) const {
        if (new_shape.numel() != numel()) {
            throw std::invalid_argument("reshape shape must have the same number of elements");
        }
        if (is_contiguous()) {
            return view(std::move(new_shape));
        }
        return clone().view(std::move(new_shape));
    }

    Tensor Tensor::transpose(size_t dim0, size_t dim1) const {
        if (dim0 >= rank() || dim1 >= rank()) {
            throw std::out_of_range("transpose dimensions out of range");
        }

        std::vector<size_t> new_dims = shape().vec();
        std::swap(new_dims[dim0], new_dims[dim1]);

        std::vector<size_t> new_strides = stride().vec();
        std::swap(new_strides[dim0], new_strides[dim1]);

        auto new_impl = std::make_shared<TensorImpl>(
            impl_->storage(), impl_->storage_offset(), Shape(new_dims), Stride(new_strides), dtype(), device()
        );
        return Tensor(new_impl);
    }

    Tensor Tensor::broadcast_to(Shape new_shape) const {
        if (shape() == new_shape) return *this;

        Stride new_stride = compute_broadcast_strides(shape(), stride(), new_shape);

        auto new_impl = std::make_shared<TensorImpl>(
            impl_->storage(), impl_->storage_offset(), std::move(new_shape), std::move(new_stride), dtype(), device()
        );
        return Tensor(new_impl);
    }

    Tensor Tensor::operator+(const Tensor& other) const { return Dispatcher::add(*this, other); }
    Tensor Tensor::operator-(const Tensor& other) const { return Dispatcher::sub(*this, other); }
    Tensor Tensor::operator*(const Tensor& other) const { return Dispatcher::mul(*this, other); }
    Tensor Tensor::operator/(const Tensor& other) const { return Dispatcher::div(*this, other); }
    Tensor Tensor::operator-() const { return Dispatcher::neg(*this); }
    Tensor Tensor::matmul(const Tensor& other) const { return Dispatcher::matmul(*this, other); }

}  // namespace helix
