#include "core/tensor.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif
#include <algorithm>  // for std::fill_n
#include <cstring>    // for memcpy
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

    auto get_autograd_provider() -> AutogradProvider* {
        if (!g_autograd_provider) {
            throw std::runtime_error("AutogradProvider has not been registered. Autograd module is not loaded.");
        }
        return g_autograd_provider;
    }

    // Factory Methods
    auto Tensor::empty(const Shape& shape) -> Tensor { return TensorFactory::empty(shape); }
    auto Tensor::zeros(const Shape& shape) -> Tensor { return TensorFactory::zeros(shape); }
    auto Tensor::ones(const Shape& shape) -> Tensor { return TensorFactory::ones(shape); }
    auto Tensor::full(const Shape& shape, const float value) -> Tensor { return TensorFactory::full(shape, value); }
    auto Tensor::randn(const Shape& shape) -> Tensor { return TensorFactory::randn(shape); }

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

    auto Tensor::shape() const -> const Shape& { return impl_->shape(); }
    auto Tensor::stride() const -> const Stride& { return impl_->stride(); }
    auto Tensor::dtype() const -> DType { return impl_->dtype(); }
    auto Tensor::device() const -> Device { return impl_->device(); }
    auto Tensor::numel() const -> size_t { return impl_->shape().numel(); }
    auto Tensor::rank() const -> size_t { return impl_->shape().rank(); }

    auto Tensor::item() const -> float {
        if (numel() != 1) {
            throw std::runtime_error("item() can only be called on tensors with 1 element");
        }
        float result = 0.0f;
        HELIX_DISPATCH_ALL_TYPES(dtype(), "item", [&] { result = static_cast<float>(data_ptr<scalar_t>()[0]); });
        return result;
    }

    auto Tensor::version() const -> uint32_t { return impl_->storage()->version(); }
    void Tensor::increment_version() { impl_->storage()->increment_version(); }

    auto Tensor::item(const std::vector<size_t>& indices) const -> float {
        size_t offset = stride().compute_offset(indices);
        float result = 0.0f;
        HELIX_DISPATCH_ALL_TYPES(dtype(), "item", [&] { result = static_cast<float>(data_ptr<scalar_t>()[offset]); });
        return result;
    }

    void Tensor::set_item(const std::vector<size_t>& indices, const float value) {
        increment_version();
        size_t offset = stride().compute_offset(indices);
        HELIX_DISPATCH_ALL_TYPES(dtype(), "set_item", [&] {
            data_ptr<scalar_t>()[offset] = static_cast<scalar_t>(value);
        });
    }

    auto Tensor::is_contiguous() const -> bool { return impl_->is_contiguous(); }

    auto Tensor::is_shared() const -> bool { return impl_.use_count() > 1 || impl_->storage().use_count() > 1; }

    auto Tensor::view(Shape new_shape) const -> Tensor { return Dispatcher::view(*this, std::move(new_shape)); }

    auto Tensor::clone() const -> Tensor { return Dispatcher::clone(*this); }

    void Tensor::copy_(const Tensor& src) {
        if (numel() != src.numel()) {
            throw std::invalid_argument("Size mismatch in copy_");
        }
        if (impl_->data() == src.impl()->data() && stride() == src.stride() && shape() == src.shape()) {
            return;
        }

        if (has_internal_overlap()) {
            throw std::runtime_error(
                "copy_: in-place operation on a tensor with overlapping memory (stride 0) is not supported."
            );
        }

        Tensor safe_src = (src.dtype() != dtype()) ? Dispatcher::cast(src, dtype()) : src;

        bool has_overlap =
            has_internal_overlap() || safe_src.has_internal_overlap() ||
            (impl_->data() != safe_src.impl()->data() || stride() != safe_src.stride() || shape() != safe_src.shape());

        if (is_contiguous() && safe_src.is_contiguous() && !has_overlap) {
            // Both are contiguous and no overlap, safe to memcpy
            std::memcpy(impl_->data(), safe_src.impl()->data(), numel() * dtype_size(dtype()));
        } else if (rank() == 2 && shape() == safe_src.shape()) {
            HELIX_DISPATCH_ALL_TYPES(dtype(), "copy_", [&] {
                scalar_t* dst_data = data_ptr<scalar_t>();
                const scalar_t* src_data = safe_src.data_ptr<scalar_t>();

                // Handle single dimension overlapping
                if (rank() == 1) {
                    size_t dst_stride = stride()[0];
                    size_t src_stride = safe_src.stride()[0];
                    for (size_t i = 0; i < shape()[0]; ++i) {
                        dst_data[i * dst_stride] = src_data[i * src_stride];
                    }
                } else {
                    // N-dimensional iterator approach
                    BinaryNDIterator it(shape());
                    it.init_from_flat(0);
                    ptrdiff_t offset_dst = it.compute_offset(stride());
                    ptrdiff_t offset_src = it.compute_offset(safe_src.stride());
                    for (size_t i = 0; i < shape().numel(); ++i) {
                        dst_data[offset_dst] = src_data[offset_src];
                        it.advance(offset_dst, stride(), offset_src, safe_src.stride());
                    }
                }
            });
        } else {
            float* dst_data = data_ptr();
            const float* src_data = safe_src.data_ptr();
            const size_t total_elements = numel();

#pragma omp parallel
            {
#if defined(_OPENMP)
                const size_t tid = omp_get_thread_num();
                const size_t num_threads = omp_get_num_threads();
#else
                const size_t tid = 0;
                const size_t num_threads = 1;
#endif
                const size_t chunk = (total_elements + num_threads - 1) / num_threads;
                const size_t start = tid * chunk;
                const size_t end = std::min(start + chunk, total_elements);

                if (start < end) {
                    NDIterator it_src(safe_src.shape());
                    NDIterator it_dst(shape());
                    it_src.init_from_flat(start);
                    it_dst.init_from_flat(start);
                    ptrdiff_t offset_src = it_src.compute_offset(safe_src.stride());
                    ptrdiff_t offset_dst = it_dst.compute_offset(stride());

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
            if (dtype() == DType::Float32 || dtype() == DType::Int32 || dtype() == DType::Float64 ||
                dtype() == DType::Int64) {
                std::memset(impl_->data(), 0, numel() * dtype_size(dtype()));
            }
        } else if (rank() == 2) {
            const size_t rows = shape()[0];
            const size_t cols = shape()[1];
            const ptrdiff_t dst_stride0 = stride()[0];
            const ptrdiff_t dst_stride1 = stride()[1];
            float* dst_data = data_ptr();

#pragma omp parallel for
            for (ptrdiff_t r = 0; r < static_cast<ptrdiff_t>(rows); ++r) {
#pragma omp simd
                for (size_t c = 0; c < cols; ++c) {
                    dst_data[r * dst_stride0 + c * dst_stride1] = 0.0f;
                }
            }
        } else {
            const size_t total_elements = numel();

#pragma omp parallel
            {
#if defined(_OPENMP)
                const size_t tid = omp_get_thread_num();
                const size_t num_threads = omp_get_num_threads();
#else
                const size_t tid = 0;
                const size_t num_threads = 1;
#endif
                const size_t chunk = (total_elements + num_threads - 1) / num_threads;
                const size_t start = tid * chunk;
                const size_t end = std::min(start + chunk, total_elements);

                if (start < end) {
                    NDIterator it(shape());
                    HELIX_DISPATCH_ALL_TYPES(dtype(), "zero_", [&] {
                        scalar_t* dst_data = data_ptr<scalar_t>();
                        if (rank() == 1) {
                            size_t dst_stride = stride()[0];
                            for (size_t i = 0; i < shape()[0]; ++i) {
                                dst_data[i * dst_stride] = 0;
                            }
                        } else {
                            NDIterator it(shape());
                            it.init_from_flat(0);
                            ptrdiff_t offset_dst = it.compute_offset(stride());
                            for (size_t i = 0; i < shape().numel(); ++i) {
                                dst_data[offset_dst] = 0;
                                it.advance(offset_dst, stride());
                            }
                        }
                    });
                }
            }
        }

        increment_version();
    }

    auto Tensor::contiguous() const -> Tensor {
        if (is_contiguous()) return *this;
        return clone();
    }

    auto Tensor::has_internal_overlap() const -> bool {
        if (rank() == 0 || numel() <= 1) return false;

        std::vector<std::pair<size_t, size_t>> stride_shape;
        for (size_t i = 0; i < rank(); ++i) {
            if (shape()[i] > 1) {
                if (stride()[i] == 0) return true;
                stride_shape.push_back({static_cast<size_t>(std::abs(stride()[i])), shape()[i]});
            }
        }

        if (stride_shape.empty()) return false;

        std::ranges::sort(stride_shape, [](const auto& a, const auto& b) { return a.first < b.first; });

        for (size_t i = 0; i < stride_shape.size() - 1; ++i) {
            if (stride_shape[i + 1].first < stride_shape[i].first * stride_shape[i].second) {
                return true;
            }
        }

        return false;
    }

    auto Tensor::reshape(Shape new_shape) const -> Tensor {
        if (new_shape.numel() != numel()) {
            throw std::invalid_argument("reshape shape must have the same number of elements");
        }
        if (is_contiguous()) {
            return view(std::move(new_shape));
        }
        return clone().view(std::move(new_shape));
    }

    auto Tensor::flatten() const -> Tensor { return reshape(Shape{numel()}); }

    auto Tensor::detach() const -> Tensor {
        // Detach creates a new Tensor that shares storage but has no autograd history.
        // It has a new TensorImpl with autograd_meta_ initialized to nullptr.
        const auto new_impl = std::make_shared<TensorImpl>(
            impl_->storage(), impl_->storage_offset(), shape(), stride(), dtype(), device()
        );
        return Tensor(new_impl);
    }

    auto Tensor::slice(const size_t dim, const size_t start, const size_t end) const -> Tensor {
        return Dispatcher::slice(*this, dim, start, end);
    }

    auto Tensor::transpose(const size_t dim0, const size_t dim1) const -> Tensor {
        return Dispatcher::transpose(*this, dim0, dim1);
    }

    auto Tensor::broadcast_to(Shape new_shape) const -> Tensor {
        return Dispatcher::broadcast_to(*this, std::move(new_shape));
    }

    auto Tensor::broadcast_to_view(Shape new_shape) const -> Tensor {
        if (shape() == new_shape) return *this;

        Stride new_stride = compute_broadcast_strides(shape(), stride(), new_shape);

        const auto new_impl = std::make_shared<TensorImpl>(
            impl_->storage(), impl_->storage_offset(), std::move(new_shape), std::move(new_stride), dtype(), device()
        );
        return Tensor(new_impl);
    }

    auto Tensor::operator+(const Tensor& other) const -> Tensor { return Dispatcher::add(*this, other); }
    auto Tensor::add_(const Tensor& other) -> Tensor& {
        Dispatcher::add_(*this, other);
        increment_version();
        return *this;
    }
    auto Tensor::operator-(const Tensor& other) const -> Tensor { return Dispatcher::sub(*this, other); }
    auto Tensor::operator*(const Tensor& other) const -> Tensor { return Dispatcher::mul(*this, other); }
    auto Tensor::operator/(const Tensor& other) const -> Tensor { return Dispatcher::div(*this, other); }

    auto Tensor::operator+(const float scalar) const -> Tensor { return Dispatcher::add_scalar(*this, scalar); }
    auto Tensor::operator-(const float scalar) const -> Tensor { return Dispatcher::sub_scalar(*this, scalar); }
    auto Tensor::operator*(const float scalar) const -> Tensor { return Dispatcher::mul_scalar(*this, scalar); }
    auto Tensor::operator/(const float scalar) const -> Tensor { return Dispatcher::div_scalar(*this, scalar); }
    auto Tensor::operator-() const -> Tensor { return Dispatcher::neg(*this); }
    auto Tensor::exp() const -> Tensor { return Dispatcher::exp(*this); }
    auto Tensor::tanh() const -> Tensor { return Dispatcher::tanh(*this); }
    auto Tensor::log() const -> Tensor { return Dispatcher::log(*this); }
    auto Tensor::sqrt() const -> Tensor { return Dispatcher::sqrt(*this); }
    auto Tensor::relu() const -> Tensor { return Dispatcher::relu(*this); }
    auto Tensor::pow(const float exponent) const -> Tensor { return Dispatcher::pow(*this, exponent); }
    auto Tensor::matmul(const Tensor& other) const -> Tensor { return Dispatcher::matmul(*this, other); }

    auto Tensor::sum(const std::optional<size_t> axis, const bool keepdim) const -> Tensor {
        return Dispatcher::sum(*this, axis, keepdim);
    }
    auto Tensor::mean(const std::optional<size_t> axis, const bool keepdim) const -> Tensor {
        return Dispatcher::mean(*this, axis, keepdim);
    }

    // Autograd API implementations
    auto Tensor::requires_grad() const -> bool { return impl_->autograd_meta() != nullptr; }

    auto Tensor::has_grad() const -> bool {
        if (!requires_grad()) return false;
        return get_autograd_provider()->has_grad(*this);
    }

    void Tensor::set_requires_grad(const bool req) const {
        if (req && (dtype() == DType::Int32 || dtype() == DType::Int64)) {
            throw std::runtime_error("Only floating point tensors can require gradients");
        }
        if (req && !requires_grad()) {
            // Lazy allocation: only create if it doesn't exist and req is true
            impl_->set_autograd_meta(get_autograd_provider()->create_meta());
        } else if (!req && requires_grad()) {
            // If setting to false, free the meta
            impl_->set_autograd_meta(nullptr);
        }
    }

    auto Tensor::grad() -> Tensor& {
        if (!requires_grad()) throw std::runtime_error("Tensor does not require grad");
        return get_autograd_provider()->get_grad(*this);
    }

    auto Tensor::grad() const -> const Tensor& {
        if (!requires_grad()) throw std::runtime_error("Tensor does not require grad");
        return get_autograd_provider()->get_grad(*this);
    }

    void Tensor::backward(const std::vector<Tensor>& grad_outputs, bool retain_graph) {
        if (!requires_grad()) throw std::runtime_error("Cannot backward on a tensor that does not require grad");
        const auto provider = get_autograd_provider();
        if (!provider) {
            throw std::runtime_error("Autograd system is not initialized.");
        }
        provider->backward(*this, grad_outputs, retain_graph);
    }

}  // namespace helix
