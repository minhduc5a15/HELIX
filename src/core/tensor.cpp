#include "core/tensor.hpp"

#include <cstddef>

#include "core/math_utils.hpp"

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
    auto Tensor::empty(const Shape& shape, std::optional<DType> dtype, std::optional<Device> device) -> Tensor {
        return TensorFactory::empty(shape, dtype, device);
    }
    auto Tensor::zeros(const Shape& shape, std::optional<DType> dtype, std::optional<Device> device) -> Tensor {
        return TensorFactory::zeros(shape, dtype, device);
    }
    auto Tensor::ones(const Shape& shape, std::optional<DType> dtype, std::optional<Device> device) -> Tensor {
        return TensorFactory::ones(shape, dtype, device);
    }
    auto Tensor::full(const Shape& shape, const float value, std::optional<DType> dtype, std::optional<Device> device)
        -> Tensor {
        return TensorFactory::full(shape, value, dtype, device);
    }
    auto Tensor::randn(const Shape& shape, std::optional<DType> dtype, std::optional<Device> device) -> Tensor {
        return TensorFactory::randn(shape, dtype, device);
    }
    void Tensor::manual_seed(uint64_t seed) { TensorFactory::manual_seed(seed); }

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

    auto Tensor::version() const -> uint32_t {
        if (!impl_ || !impl_->storage()) return 0;
        return impl_->storage()->version();
    }

    void Tensor::increment_version() {
        if (impl_ && impl_->storage()) {
            impl_->storage()->increment_version();
        }
    }

    auto Tensor::item(const std::vector<size_t>& indices) const -> float {
        if (indices.size() != rank()) {
            throw std::invalid_argument("Indices rank must match tensor rank");
        }
        for (size_t i = 0; i < rank(); ++i) {
            if (indices[i] >= shape()[i]) {
                throw std::out_of_range("Tensor index out of range for dimension " + std::to_string(i));
            }
        }
        ptrdiff_t offset = stride().compute_offset(indices);
        float result = 0.0f;
        HELIX_DISPATCH_ALL_TYPES(dtype(), "item", [&] { result = static_cast<float>(data_ptr<scalar_t>()[offset]); });
        return result;
    }

    namespace {
        inline void check_inplace_autograd_invariance(const Tensor& target, const Tensor* source = nullptr) {
            if (Dispatcher::get_graph_builder() && (target.requires_grad() || (source && source->requires_grad()))) {
                if (target.requires_grad()) {
                    if (target.is_leaf()) {
                        throw std::runtime_error(
                            "a leaf Tensor that requires grad is being used in an in-place operation."
                        );
                    }
                    throw std::runtime_error(
                        "in-place operations on non-leaf tensors that require grad are currently not supported in "
                        "HELIX "
                        "Autograd."
                    );
                }
                throw std::runtime_error(
                    "in-place operations involving tensors that require grad are currently not supported in HELIX "
                    "Autograd."
                );
            }
        }
    }  // namespace

    void Tensor::set_item(const std::vector<size_t>& indices, const float value) {
        check_inplace_autograd_invariance(*this);
        if (indices.size() != rank()) {
            throw std::invalid_argument("Indices rank must match tensor rank");
        }
        for (size_t i = 0; i < rank(); ++i) {
            if (indices[i] >= shape()[i]) {
                throw std::out_of_range("Tensor index out of range for dimension " + std::to_string(i));
            }
        }
        increment_version();
        ptrdiff_t offset = stride().compute_offset(indices);
        HELIX_DISPATCH_ALL_TYPES(dtype(), "set_item", [&] {
            data_ptr<scalar_t>()[offset] = static_cast<scalar_t>(value);
        });
    }

    auto Tensor::is_contiguous() const -> bool { return impl_->is_contiguous(); }

    auto Tensor::is_shared() const -> bool { return impl_.use_count() > 1 || impl_->storage().use_count() > 1; }

    auto Tensor::view(Shape new_shape) const -> Tensor { return Dispatcher::view(*this, std::move(new_shape)); }

    auto Tensor::clone() const -> Tensor { return Dispatcher::clone(*this); }

    namespace {
        template <typename scalar_t>
        void copy_fallback_kernel(
            scalar_t* dst_data, const scalar_t* src_data, const Tensor& dst, const Tensor& safe_src
        ) {
            if (dst.rank() > 8 || safe_src.rank() > 8) {
                throw std::invalid_argument("Operation on non-contiguous tensor with rank > 8 is not supported");
            }
            const size_t total_elements = dst.numel();

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
                    NDIterator it_dst(dst.shape());
                    it_src.init_from_flat(start);
                    it_dst.init_from_flat(start);
                    ptrdiff_t offset_src = it_src.compute_offset(safe_src.stride());
                    ptrdiff_t offset_dst = it_dst.compute_offset(dst.stride());

                    for (size_t i = start; i < end; ++i) {
                        dst_data[offset_dst] = src_data[offset_src];
                        it_src.advance(offset_src, safe_src.stride());
                        it_dst.advance(offset_dst, dst.stride());
                    }
                }
            }
        }

        template <typename scalar_t>
        void zero_2d_kernel(
            scalar_t* dst_data, size_t rows, size_t cols, ptrdiff_t dst_stride0, ptrdiff_t dst_stride1
        ) {
#pragma omp parallel for
            for (ptrdiff_t r = 0; r < static_cast<ptrdiff_t>(rows); ++r) {
#pragma omp simd
                for (size_t c = 0; c < cols; ++c) {
                    dst_data[r * dst_stride0 + static_cast<ptrdiff_t>(c) * dst_stride1] = static_cast<scalar_t>(0);
                }
            }
        }
    }  // namespace

    void Tensor::copy_(const Tensor& src) {
        check_inplace_autograd_invariance(*this, &src);
        if (numel() != src.numel()) {
            throw std::invalid_argument("Size mismatch in copy_");
        }
        if (impl_->data() == src.impl()->data() && stride() == src.stride() && shape() == src.shape()) {
            return;
        }

        if (numel() == 0 || src.numel() == 0) {
            return;
        }

        if (has_internal_overlap()) {
            throw std::runtime_error(
                "copy_: in-place operation on a tensor with overlapping memory (stride 0) is not supported."
            );
        }

        Tensor safe_src = (src.dtype() != dtype()) ? Dispatcher::cast(src, dtype()) : src;

        bool is_aliased = (impl_->storage() == safe_src.impl()->storage());
        if (is_aliased) {
            safe_src = safe_src.clone();
        }
        bool has_overlap = has_internal_overlap() || safe_src.has_internal_overlap();

        if (is_contiguous() && safe_src.is_contiguous() && !has_overlap) {
            // Both are contiguous and no overlap, safe to memcpy
            std::memcpy(impl_->data(), safe_src.impl()->data(), numel() * dtype_size(dtype()));
        } else if (shape() == safe_src.shape()) {
            HELIX_DISPATCH_ALL_TYPES(dtype(), "copy_", [&] {
                scalar_t* dst_data = data_ptr<scalar_t>();
                const scalar_t* src_data = safe_src.data_ptr<scalar_t>();

                // Handle single dimension overlapping
                if (rank() == 1) {
                    ptrdiff_t dst_stride = stride()[0];
                    ptrdiff_t src_stride = safe_src.stride()[0];
                    for (size_t i = 0; i < shape()[0]; ++i) {
                        dst_data[i * dst_stride] = src_data[i * src_stride];
                    }
                } else {
                    if (shape().rank() > 8) {
                        throw std::invalid_argument("Operation on non-contiguous tensor with rank > 8 is not supported"
                        );
                    }
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
            HELIX_DISPATCH_ALL_TYPES(dtype(), "copy_", [&] {
                scalar_t* dst_data = data_ptr<scalar_t>();
                const scalar_t* src_data = safe_src.data_ptr<scalar_t>();
                copy_fallback_kernel<scalar_t>(dst_data, src_data, *this, safe_src);
            });
        }

        increment_version();
    }

    void Tensor::zero_() {
        check_inplace_autograd_invariance(*this);
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

            HELIX_DISPATCH_ALL_TYPES(dtype(), "zero_2d", [&] {
                scalar_t* dst_data = data_ptr<scalar_t>();
                zero_2d_kernel<scalar_t>(dst_data, rows, cols, dst_stride0, dst_stride1);
            });
        } else {
            if (rank() > 8) {
                throw std::invalid_argument("Operation on non-contiguous tensor with rank > 8 is not supported");
            }
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
                    HELIX_DISPATCH_ALL_TYPES(dtype(), "zero_", [&] {
                        scalar_t* dst_data = data_ptr<scalar_t>();
                        if (rank() == 1) {
                            const ptrdiff_t dst_stride = stride()[0];
                            for (size_t i = start; i < end; ++i) {
                                dst_data[static_cast<ptrdiff_t>(i) * dst_stride] = static_cast<scalar_t>(0);
                            }
                        } else {
                            NDIterator it(shape());
                            it.init_from_flat(start);
                            ptrdiff_t offset_dst = it.compute_offset(stride());
                            for (size_t i = start; i < end; ++i) {
                                dst_data[offset_dst] = static_cast<scalar_t>(0);
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
        // No element, no drama.
        if (rank() == 0 || numel() <= 1) return false;

        std::vector<std::pair<size_t, size_t>> stride_shape;
        for (size_t i = 0; i < rank(); ++i) {
            if (shape()[i] > 1) {
                if (stride()[i] == 0) return true;  // sharing is not caring here
                stride_shape.push_back({static_cast<size_t>(std::abs(stride()[i])), shape()[i]});
            }
        }
        if (stride_shape.empty()) return false;

        std::ranges::sort(stride_shape, [](auto& a, auto& b) { return a.first < b.first; });

        for (size_t i = 0; i < stride_shape.size() - 1; ++i) {
            // Conservative? Yes. Paranoid? Maybe. Safe? Absolutely.
            // This may return true for perfectly innocent tensors (false positive).
            // But we'd rather clone a few extra times than corrupt memory silently.
            // Life is too short for exact integer linear programming here.
            // (And PyTorch does it too, so if we're wrong, we're in good company.)
            size_t span = 0;
            if (mul_overflow(stride_shape[i].first, stride_shape[i].second, &span) ||
                stride_shape[i + 1].first < span) {
                return true;
            }
        }
        // We're pretty sure there's no overlap. If there is, well, it's not our fault.
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

    Tensor flatten(const Tensor& input, int64_t start_dim, int64_t end_dim) {
        return input.flatten(start_dim, end_dim);
    }

    Tensor unsqueeze(const Tensor& input, int64_t dim) { return input.unsqueeze(dim); }

    Tensor squeeze(const Tensor& input) { return input.squeeze(); }

    Tensor squeeze(const Tensor& input, int64_t dim) { return input.squeeze(dim); }

    Tensor bmm(const Tensor& input, const Tensor& mat2) { return input.bmm(mat2); }

    auto Tensor::flatten(int64_t start_dim, int64_t end_dim) const -> Tensor {
        const int64_t r = static_cast<int64_t>(rank());

        if (r == 0) {
            if ((start_dim == 0 || start_dim == -1) && (end_dim == 0 || end_dim == -1)) {
                return reshape(Shape{1});
            }
            throw std::out_of_range("flatten dimension out of range for scalar tensor (expected 0 or -1)");
        }

        const int64_t norm_start = start_dim < 0 ? start_dim + r : start_dim;
        const int64_t norm_end = end_dim < 0 ? end_dim + r : end_dim;

        if (norm_start < 0 || norm_start >= r) {
            throw std::out_of_range("flatten start_dim out of range: " + std::to_string(start_dim));
        }
        if (norm_end < 0 || norm_end >= r) {
            throw std::out_of_range("flatten end_dim out of range: " + std::to_string(end_dim));
        }
        if (norm_start > norm_end) {
            throw std::invalid_argument(
                "flatten: start_dim (" + std::to_string(start_dim) + ") cannot come after end_dim (" +
                std::to_string(end_dim) + ")"
            );
        }

        if (norm_start == norm_end) {
            return *this;
        }

        std::vector<size_t> new_dims;
        new_dims.reserve(static_cast<size_t>(r - (norm_end - norm_start)));

        for (int64_t i = 0; i < norm_start; ++i) {
            new_dims.push_back(shape()[static_cast<size_t>(i)]);
        }

        size_t collapsed = 1;
        for (int64_t i = norm_start; i <= norm_end; ++i) {
            const size_t dim_size = shape()[static_cast<size_t>(i)];
            if (mul_overflow(collapsed, dim_size, &collapsed)) {
                throw std::overflow_error("flatten dimension product overflowed size_t");
            }
        }
        new_dims.push_back(collapsed);

        for (int64_t i = norm_end + 1; i < r; ++i) {
            new_dims.push_back(shape()[static_cast<size_t>(i)]);
        }

        return reshape(Shape(std::move(new_dims)));
    }

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

    auto Tensor::unsqueeze(int64_t dim) const -> Tensor { return Dispatcher::unsqueeze(*this, dim); }

    auto Tensor::squeeze() const -> Tensor { return Dispatcher::squeeze(*this); }

    auto Tensor::squeeze(int64_t dim) const -> Tensor { return Dispatcher::squeeze(*this, dim); }

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
    auto Tensor::bmm(const Tensor& other) const -> Tensor { return Dispatcher::bmm(*this, other); }

    auto Tensor::sum(const std::optional<size_t> axis, const bool keepdim) const -> Tensor {
        return Dispatcher::sum(*this, axis, keepdim);
    }
    auto Tensor::mean(const std::optional<size_t> axis, const bool keepdim) const -> Tensor {
        return Dispatcher::mean(*this, axis, keepdim);
    }
    auto Tensor::argmax(size_t dim) const -> Tensor { return Dispatcher::argmax(*this, dim); }

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
            impl_->set_autograd_meta(get_autograd_provider()->create_meta(dtype()));
        } else if (!req && requires_grad()) {
            // If setting to false, free the meta
            impl_->set_autograd_meta(nullptr);
        }
    }

    auto Tensor::is_leaf() const -> bool {
        if (!requires_grad()) return true;
        auto* provider = get_autograd_provider();
        if (!provider) return true;
        return provider->is_leaf(*this);
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
