#include "core/tensor.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif
#include <algorithm>  // for std::fill_n
#include <algorithm>
#include <cstring>  // for memcpy
#include <stdexcept>

#include "core/autograd_meta.hpp"
#include "core/broadcast.hpp"
#include "core/dispatcher.hpp"
#include "core/nd_iterator.hpp"
#include "core/tensor_factory.hpp"

namespace helix {

    // Autograd Provider Registry
    static AutogradProvider* g_autograd_provider = nullptr;

    void register_autograd_provider(AutogradProvider* provider) { g_autograd_provider = provider; }

    AutogradProvider* get_autograd_provider() {
        if (!g_autograd_provider) {
            throw std::runtime_error("AutogradProvider has not been registered. Autograd module is not loaded.");
        }
        return g_autograd_provider;
    }

    // Factory Methods
    Tensor Tensor::empty(const Shape& shape) { return TensorFactory::empty(shape); }
    Tensor Tensor::zeros(const Shape& shape) { return TensorFactory::zeros(shape); }
    Tensor Tensor::ones(const Shape& shape) { return TensorFactory::ones(shape); }
    Tensor Tensor::full(const Shape& shape, const float value) { return TensorFactory::full(shape, value); }
    Tensor Tensor::randn(const Shape& shape) { return TensorFactory::randn(shape); }

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

    uint32_t Tensor::version() const { return impl_->storage()->version(); }
    void Tensor::increment_version() { impl_->storage()->increment_version(); }

    float Tensor::item(const std::vector<size_t>& indices) const {
        const size_t offset = stride().compute_offset(indices);
        return data_ptr()[offset];
    }

    void Tensor::set_item(const std::vector<size_t>& indices, const float value) {
        const size_t offset = stride().compute_offset(indices);
        data_ptr()[offset] = value;
    }

    bool Tensor::is_contiguous() const { return impl_->is_contiguous(); }

    bool Tensor::is_shared() const { return impl_.use_count() > 1 || impl_->storage().use_count() > 1; }

    Tensor Tensor::view(Shape new_shape) const {
        if (new_shape.numel() != numel()) {
            throw std::invalid_argument("view shape must have the same number of elements");
        }
        if (!is_contiguous()) {
            throw std::runtime_error("view cannot be called on non-contiguous tensor, use reshape instead");
        }
        const auto new_impl = std::make_shared<TensorImpl>(
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
            if (numel() > 0) {
                std::memcpy(new_tensor.data_ptr(), data_ptr(), numel() * sizeof(float));
            }
        } else if (rank() == 2) {
            const size_t rows = shape()[0];
            const size_t cols = shape()[1];
            const size_t src_stride0 = stride()[0];
            const size_t src_stride1 = stride()[1];
            const size_t dst_stride0 = new_tensor.stride()[0];
            const size_t dst_stride1 = new_tensor.stride()[1];
            float* dst_data = new_tensor.data_ptr();
            const float* src_data = data_ptr();

#pragma omp parallel for
            for (ptrdiff_t r = 0; r < static_cast<ptrdiff_t>(rows); ++r) {
#pragma omp simd
                for (size_t c = 0; c < cols; ++c) {
                    dst_data[r * dst_stride0 + c * dst_stride1] = src_data[r * src_stride0 + c * src_stride1];
                }
            }
        } else {
            float* dst_data = new_tensor.data_ptr();
            const float* src_data = data_ptr();
            size_t total_elements = numel();

#pragma omp parallel
            {
#if defined(_OPENMP)
                size_t tid = omp_get_thread_num();
                size_t num_threads = omp_get_num_threads();
#else
                size_t tid = 0;
                size_t num_threads = 1;
#endif
                size_t chunk = (total_elements + num_threads - 1) / num_threads;
                size_t start = tid * chunk;
                size_t end = std::min(start + chunk, total_elements);

                if (start < end) {
                    BinaryNDIterator it(shape());
                    it.init_from_flat(start);
                    size_t offset_src = it.compute_offset(stride());
                    size_t offset_dst = it.compute_offset(new_tensor.stride());

                    for (size_t i = start; i < end; ++i) {
                        dst_data[offset_dst] = src_data[offset_src];
                        it.advance(offset_src, stride(), offset_dst, new_tensor.stride());
                    }
                }
            }
        }
        return new_tensor;
    }

    void Tensor::copy_(const Tensor& src) {
        if (numel() != src.numel()) {
            throw std::invalid_argument("Tensor sizes do not match for copy_");
        }

        if (data_ptr() == src.data_ptr() && stride() == src.stride() && shape() == src.shape()) {
            return;
        }

        if (has_internal_overlap()) {
            throw std::runtime_error(
                "copy_: in-place operation on a tensor with overlapping memory (stride 0) is not supported."
            );
        }

        bool is_aliased = (impl_->storage() == src.impl_->storage()) &&
                          (data_ptr() != src.data_ptr() || stride() != src.stride() || shape() != src.shape());
        Tensor safe_src = is_aliased ? src.clone() : src;

        if (is_contiguous() && safe_src.is_contiguous()) {
            if (numel() > 0) {
                std::memcpy(data_ptr(), safe_src.data_ptr(), numel() * sizeof(float));
            }
        } else if (rank() == 2 && shape() == safe_src.shape()) {
            const size_t rows = shape()[0];
            const size_t cols = shape()[1];
            const size_t dst_stride0 = stride()[0];
            const size_t dst_stride1 = stride()[1];
            const size_t src_stride0 = safe_src.stride()[0];
            const size_t src_stride1 = safe_src.stride()[1];
            float* dst_data = data_ptr();
            const float* src_data = safe_src.data_ptr();

#pragma omp parallel for
            for (ptrdiff_t r = 0; r < static_cast<ptrdiff_t>(rows); ++r) {
#pragma omp simd
                for (size_t c = 0; c < cols; ++c) {
                    dst_data[r * dst_stride0 + c * dst_stride1] = src_data[r * src_stride0 + c * src_stride1];
                }
            }
        } else {
            float* dst_data = data_ptr();
            const float* src_data = safe_src.data_ptr();
            size_t total_elements = numel();

#pragma omp parallel
            {
#if defined(_OPENMP)
                size_t tid = omp_get_thread_num();
                size_t num_threads = omp_get_num_threads();
#else
                size_t tid = 0;
                size_t num_threads = 1;
#endif
                size_t chunk = (total_elements + num_threads - 1) / num_threads;
                size_t start = tid * chunk;
                size_t end = std::min(start + chunk, total_elements);

                if (start < end) {
                    NDIterator it_src(safe_src.shape());
                    NDIterator it_dst(shape());
                    it_src.init_from_flat(start);
                    it_dst.init_from_flat(start);
                    size_t offset_src = it_src.compute_offset(safe_src.stride());
                    size_t offset_dst = it_dst.compute_offset(stride());

                    for (size_t i = start; i < end; ++i) {
                        dst_data[offset_dst] = src_data[offset_src];
                        it_src.advance(offset_src, safe_src.stride());
                        it_dst.advance(offset_dst, stride());
                    }
                }
            }
        }

        increment_version();
    }

    void Tensor::zero_() {
        if (has_internal_overlap()) {
            throw std::runtime_error(
                "zero_: in-place operation on a tensor with overlapping memory (stride 0) is not supported."
            );
        }

        if (is_contiguous()) {
            if (numel() > 0) {
                std::fill_n(data_ptr(), numel(), 0.0f);
            }
        } else if (rank() == 2) {
            const size_t rows = shape()[0];
            const size_t cols = shape()[1];
            const size_t dst_stride0 = stride()[0];
            const size_t dst_stride1 = stride()[1];
            float* dst_data = data_ptr();

#pragma omp parallel for
            for (ptrdiff_t r = 0; r < static_cast<ptrdiff_t>(rows); ++r) {
#pragma omp simd
                for (size_t c = 0; c < cols; ++c) {
                    dst_data[r * dst_stride0 + c * dst_stride1] = 0.0f;
                }
            }
        } else {
            float* dst_data = data_ptr();
            size_t total_elements = numel();

#pragma omp parallel
            {
#if defined(_OPENMP)
                size_t tid = omp_get_thread_num();
                size_t num_threads = omp_get_num_threads();
#else
                size_t tid = 0;
                size_t num_threads = 1;
#endif
                size_t chunk = (total_elements + num_threads - 1) / num_threads;
                size_t start = tid * chunk;
                size_t end = std::min(start + chunk, total_elements);

                if (start < end) {
                    NDIterator it(shape());
                    it.init_from_flat(start);
                    size_t offset_dst = it.compute_offset(stride());

                    for (size_t i = start; i < end; ++i) {
                        dst_data[offset_dst] = 0.0f;
                        it.advance(offset_dst, stride());
                    }
                }
            }
        }

        increment_version();
    }

    Tensor Tensor::contiguous() const {
        if (is_contiguous()) return *this;
        return clone();
    }

    bool Tensor::has_internal_overlap() const {
        for (size_t i = 0; i < rank(); ++i) {
            if (stride()[i] == 0 && shape()[i] > 1) {
                return true;
            }
        }
        return false;
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

    Tensor Tensor::flatten() const { return reshape(Shape{numel()}); }

    Tensor Tensor::detach() const {
        // Detach creates a new Tensor that shares storage but has no autograd history.
        // It has a new TensorImpl with autograd_meta_ initialized to nullptr.
        const auto new_impl = std::make_shared<TensorImpl>(
            impl_->storage(), impl_->storage_offset(), shape(), stride(), dtype(), device()
        );
        return Tensor(new_impl);
    }

    Tensor Tensor::slice(const size_t dim, const size_t start, const size_t end) const {
        if (dim >= rank()) {
            throw std::out_of_range("slice dimension out of range");
        }
        if (start >= end || end > shape()[dim]) {
            throw std::invalid_argument("invalid slice bounds");
        }

        std::vector<size_t> new_dims = shape().vec();
        new_dims[dim] = end - start;

        size_t new_offset = impl_->storage_offset() + start * stride()[dim];

        const auto new_impl =
            std::make_shared<TensorImpl>(impl_->storage(), new_offset, Shape(new_dims), stride(), dtype(), device());
        return Tensor(new_impl);
    }

    Tensor Tensor::transpose(const size_t dim0, const size_t dim1) const {
        if (dim0 >= rank() || dim1 >= rank()) {
            throw std::out_of_range("transpose dimensions out of range");
        }

        std::vector<size_t> new_dims = shape().vec();
        std::swap(new_dims[dim0], new_dims[dim1]);

        std::vector<size_t> new_strides = stride().vec();
        std::swap(new_strides[dim0], new_strides[dim1]);

        const auto new_impl = std::make_shared<TensorImpl>(
            impl_->storage(), impl_->storage_offset(), Shape(new_dims), Stride(new_strides), dtype(), device()
        );
        return Tensor(new_impl);
    }

    Tensor Tensor::broadcast_to(Shape new_shape) const {
        if (shape() == new_shape) return *this;

        Stride new_stride = compute_broadcast_strides(shape(), stride(), new_shape);

        const auto new_impl = std::make_shared<TensorImpl>(
            impl_->storage(), impl_->storage_offset(), std::move(new_shape), std::move(new_stride), dtype(), device()
        );
        return Tensor(new_impl);
    }

    Tensor Tensor::operator+(const Tensor& other) const { return Dispatcher::add(*this, other); }
    Tensor& Tensor::add_(const Tensor& other) {
        Dispatcher::add_(*this, other);
        increment_version();
        return *this;
    }
    Tensor Tensor::operator-(const Tensor& other) const { return Dispatcher::sub(*this, other); }
    Tensor Tensor::operator*(const Tensor& other) const { return Dispatcher::mul(*this, other); }
    Tensor Tensor::operator/(const Tensor& other) const { return Dispatcher::div(*this, other); }

    Tensor Tensor::operator+(const float scalar) const { return Dispatcher::add_scalar(*this, scalar); }
    Tensor Tensor::operator-(const float scalar) const { return Dispatcher::sub_scalar(*this, scalar); }
    Tensor Tensor::operator*(const float scalar) const { return Dispatcher::mul_scalar(*this, scalar); }
    Tensor Tensor::operator/(const float scalar) const { return Dispatcher::div_scalar(*this, scalar); }
    Tensor Tensor::operator-() const { return Dispatcher::neg(*this); }
    Tensor Tensor::exp() const { return Dispatcher::exp(*this); }
    Tensor Tensor::log() const { return Dispatcher::log(*this); }
    Tensor Tensor::sqrt() const { return Dispatcher::sqrt(*this); }
    Tensor Tensor::relu() const { return Dispatcher::relu(*this); }
    Tensor Tensor::pow(const float exponent) const { return Dispatcher::pow(*this, exponent); }
    Tensor Tensor::matmul(const Tensor& other) const { return Dispatcher::matmul(*this, other); }

    Tensor Tensor::sum(const std::optional<size_t> axis, const bool keepdim) const {
        return Dispatcher::sum(*this, axis, keepdim);
    }
    Tensor Tensor::mean(const std::optional<size_t> axis, const bool keepdim) const {
        return Dispatcher::mean(*this, axis, keepdim);
    }

    // Autograd API implementations
    bool Tensor::requires_grad() const { return impl_->autograd_meta() != nullptr; }

    bool Tensor::has_grad() const {
        if (!requires_grad()) return false;
        return get_autograd_provider()->has_grad(*this);
    }

    void Tensor::set_requires_grad(const bool req) const {
        if (req && !requires_grad()) {
            // Lazy allocation: only create if it doesn't exist and req is true
            impl_->set_autograd_meta(get_autograd_provider()->create_meta());
        } else if (!req && requires_grad()) {
            // If setting to false, free the meta
            impl_->set_autograd_meta(nullptr);
        }
    }

    Tensor& Tensor::grad() {
        if (!requires_grad()) throw std::runtime_error("Tensor does not require grad");
        return get_autograd_provider()->get_grad(*this);
    }

    const Tensor& Tensor::grad() const {
        if (!requires_grad()) throw std::runtime_error("Tensor does not require grad");
        return get_autograd_provider()->get_grad(*this);
    }

    void Tensor::backward(const std::vector<Tensor>& grad_outputs) {
        if (!requires_grad()) throw std::runtime_error("Cannot backward on a tensor that does not require grad");
        get_autograd_provider()->backward(*this, grad_outputs);
    }

}  // namespace helix
