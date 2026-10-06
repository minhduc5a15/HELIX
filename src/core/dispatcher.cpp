#include "core/dispatcher.hpp"

#include <cstring>

#if defined(_OPENMP)
#include <omp.h>
#endif
#include <algorithm>
#include <atomic>
#include <stdexcept>

#include "backend/cpu_backend.hpp"
#include "core/broadcast.hpp"
#include "core/graph_builder.hpp"
#include "core/nd_iterator.hpp"
#include "core/tensor.hpp"
#include "core/tensor_iterator.hpp"

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define HELIX_TSAN_ENABLED 1
#endif
#elif defined(__SANITIZE_THREAD__)
#define HELIX_TSAN_ENABLED 1
#endif

#if defined(HELIX_TSAN_ENABLED)
extern "C" const char* __tsan_default_options() { return "ignore_noninstrumented_modules=1"; }
#endif

namespace helix {
    namespace {
        template <typename scalar_t, typename Operation>
        void run_binary_kernel(
            const Tensor& lhs,
            const Tensor& rhs,
            Tensor& out,
            void (*contiguous_kernel)(const scalar_t*, const scalar_t*, scalar_t*, size_t),
            Operation&& operation
        ) {
            // Keep the common dense path independent of TensorIterator's fixed-rank
            // traversal state. Besides avoiding iterator setup, this preserves support
            // for contiguous tensors whose rank exceeds MAX_TENSOR_ITERATOR_DIMS.
            if (lhs.is_contiguous() && rhs.is_contiguous() && out.is_contiguous() && lhs.shape() == out.shape() &&
                rhs.shape() == out.shape()) {
                contiguous_kernel(
                    lhs.data_ptr<scalar_t>(), rhs.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), out.numel()
                );
                return;
            }

            TensorIterator iterator(out, {&lhs, &rhs});
            if (iterator.is_contiguous()) {
                contiguous_kernel(
                    lhs.data_ptr<scalar_t>(), rhs.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), out.numel()
                );
                return;
            }

            scalar_t* output_data = out.data_ptr<scalar_t>();
            const scalar_t* lhs_data = lhs.data_ptr<scalar_t>();
            const scalar_t* rhs_data = rhs.data_ptr<scalar_t>();
            iterator.for_each_parallel([&](const size_t, const auto& offsets) {
                output_data[offsets[0]] = operation(lhs_data[offsets[1]], rhs_data[offsets[2]]);
            });
        }

        template <typename src_t, typename dst_t>
        void cast_kernel(const src_t* src_data, dst_t* dst_data, size_t numel) {
#pragma omp parallel for if (numel >= OMP_THRESHOLD)
            for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(numel); ++i) {
                dst_data[i] = static_cast<dst_t>(src_data[i]);
            }
        }

        template <typename src_t>
        void dispatch_cast_dst(DType new_dtype, const Tensor& lhs, Tensor& out) {
            HELIX_DISPATCH_ALL_TYPES(new_dtype, "cast_dst", [&] {
                using dst_t = scalar_t;
                cast_kernel<src_t, dst_t>(lhs.data_ptr<src_t>(), out.data_ptr<dst_t>(), lhs.numel());
            });
        }

        template <typename scalar_t>
        void add_inplace_kernel(Tensor& a, const Tensor& safe_b) {
            if (a.is_contiguous() && safe_b.is_contiguous()) {
                const size_t total_elements = a.numel();
                scalar_t* a_data = a.data_ptr<scalar_t>();
                const scalar_t* b_data = safe_b.data_ptr<scalar_t>();

#pragma omp parallel if (total_elements >= OMP_THRESHOLD)
                {
#if defined(_OPENMP)
                    const size_t num_threads = omp_get_num_threads();
                    const size_t tid = omp_get_thread_num();
#else
                    const size_t num_threads = 1;
                    const size_t tid = 0;
#endif
                    const size_t chunk = (total_elements + num_threads - 1) / num_threads;
                    const size_t start = tid * chunk;
                    const size_t end = std::min(start + chunk, total_elements);

                    if (start < end) {
                        CPUBackend::add(a_data + start, b_data + start, a_data + start, end - start);
                    }
                }
            } else if (a.rank() == 2) {
                const size_t rows = a.shape()[0];
                const size_t cols = a.shape()[1];
                const ptrdiff_t a_stride0 = a.stride()[0];
                const ptrdiff_t a_stride1 = a.stride()[1];
                const ptrdiff_t b_stride0 = safe_b.stride()[0];
                const ptrdiff_t b_stride1 = safe_b.stride()[1];
                scalar_t* a_data = a.data_ptr<scalar_t>();
                const scalar_t* b_data = safe_b.data_ptr<scalar_t>();

#pragma omp parallel for if (a.numel() >= OMP_THRESHOLD)
                for (ptrdiff_t r = 0; r < static_cast<ptrdiff_t>(rows); ++r) {
#pragma omp simd
                    for (ptrdiff_t c = 0; c < static_cast<ptrdiff_t>(cols); ++c) {
                        a_data[r * a_stride0 + c * a_stride1] += b_data[r * b_stride0 + c * b_stride1];
                    }
                }
            } else {
                if (a.rank() > 8 || safe_b.rank() > 8) {
                    throw std::invalid_argument("Operation on non-contiguous tensor with rank > 8 is not supported");
                }
                scalar_t* a_data = a.data_ptr<scalar_t>();
                const scalar_t* b_data = safe_b.data_ptr<scalar_t>();
                const size_t total_elements = a.numel();

#pragma omp parallel if (total_elements >= OMP_NON_CONTIGUOUS_THRESHOLD)
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
                        BinaryNDIterator it(a.shape());
                        it.init_from_flat(start);
                        ptrdiff_t offset_a = it.compute_offset(a.stride());
                        ptrdiff_t offset_b = it.compute_offset(safe_b.stride());

                        for (size_t i = start; i < end; ++i) {
                            a_data[offset_a] += b_data[offset_b];
                            it.advance(offset_a, a.stride(), offset_b, safe_b.stride());
                        }
                    }
                }
            }
        }

        template <typename scalar_t>
        void sgd_inplace_kernel(Tensor& param, const Tensor& safe_grad, const float lr) {
            if (param.is_contiguous() && safe_grad.is_contiguous()) {
                const size_t total_elements = param.numel();
                scalar_t* p_data = param.data_ptr<scalar_t>();
                const scalar_t* g_data = safe_grad.data_ptr<scalar_t>();

#pragma omp parallel if (total_elements >= OMP_THRESHOLD)
                {
#if defined(_OPENMP)
                    const size_t num_threads = omp_get_num_threads();
                    const size_t tid = omp_get_thread_num();
#else
                    const size_t num_threads = 1;
                    const size_t tid = 0;
#endif
                    const size_t chunk = (total_elements + num_threads - 1) / num_threads;
                    const size_t start = tid * chunk;
                    const size_t end = std::min(start + chunk, total_elements);

                    if (start < end) {
                        CPUBackend::sgd(p_data + start, g_data + start, lr, end - start);
                    }
                }
            } else if (param.rank() == 2) {
                const size_t rows = param.shape()[0];
                const size_t cols = param.shape()[1];
                const ptrdiff_t p_stride0 = param.stride()[0];
                const ptrdiff_t p_stride1 = param.stride()[1];
                const ptrdiff_t g_stride0 = safe_grad.stride()[0];
                const ptrdiff_t g_stride1 = safe_grad.stride()[1];
                scalar_t* p_data = param.data_ptr<scalar_t>();
                const scalar_t* g_data = safe_grad.data_ptr<scalar_t>();

#pragma omp parallel for if (param.numel() >= OMP_THRESHOLD)
                for (ptrdiff_t r = 0; r < static_cast<ptrdiff_t>(rows); ++r) {
#pragma omp simd
                    for (ptrdiff_t c = 0; c < static_cast<ptrdiff_t>(cols); ++c) {
                        p_data[r * p_stride0 + c * p_stride1] -=
                            static_cast<scalar_t>(lr) * g_data[r * g_stride0 + c * g_stride1];
                    }
                }
            } else {
                if (param.rank() > 8 || safe_grad.rank() > 8) {
                    throw std::invalid_argument("Operation on non-contiguous tensor with rank > 8 is not supported");
                }
                scalar_t* p_data = param.data_ptr<scalar_t>();
                const scalar_t* g_data = safe_grad.data_ptr<scalar_t>();
                const size_t total_elements = param.numel();

#pragma omp parallel if (total_elements >= OMP_NON_CONTIGUOUS_THRESHOLD)
                {
#if defined(_OPENMP)
                    const size_t tid = omp_get_thread_num();
                    const size_t num_threads = omp_get_num_threads();
#else
                    size_t tid = 0;
                    size_t num_threads = 1;
#endif
                    const size_t chunk = (total_elements + num_threads - 1) / num_threads;
                    const size_t start = tid * chunk;
                    const size_t end = std::min(start + chunk, total_elements);

                    if (start < end) {
                        BinaryNDIterator it(param.shape());
                        it.init_from_flat(start);
                        ptrdiff_t offset_p = it.compute_offset(param.stride());
                        ptrdiff_t offset_g = it.compute_offset(safe_grad.stride());

                        for (size_t i = start; i < end; ++i) {
                            p_data[offset_p] -= static_cast<scalar_t>(lr) * g_data[offset_g];
                            it.advance(offset_p, param.stride(), offset_g, safe_grad.stride());
                        }
                    }
                }
            }
        }
    }  // namespace

    static std::atomic<GraphBuilderInterface*> g_global_graph_builder{nullptr};
    static thread_local std::optional<GraphBuilderInterface*> tls_graph_builder_override;

    void Dispatcher::register_graph_builder(GraphBuilderInterface* builder) {
        if (builder != nullptr) {
            g_global_graph_builder.store(builder, std::memory_order_release);
        }
        tls_graph_builder_override = builder;
    }

    GraphBuilderInterface* Dispatcher::get_graph_builder() {
        if (tls_graph_builder_override.has_value()) return *tls_graph_builder_override;
        return g_global_graph_builder.load(std::memory_order_acquire);
    }

    std::optional<GraphBuilderInterface*> Dispatcher::get_thread_graph_builder_override() {
        return tls_graph_builder_override;
    }

    void Dispatcher::set_thread_graph_builder_override(std::optional<GraphBuilderInterface*> builder) {
        tls_graph_builder_override = builder;
    }

#define g_graph_builder (Dispatcher::get_graph_builder())

    template <typename scalar_t>
    void clone_impl(const Tensor& a, Tensor& new_tensor) {
        if (a.is_contiguous()) {
            if (a.numel() > 0) {
                std::memcpy(new_tensor.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), a.numel() * sizeof(scalar_t));
            }
        } else if (a.rank() == 2) {
            const size_t rows = a.shape()[0];
            const size_t cols = a.shape()[1];
            const ptrdiff_t src_stride0 = a.stride()[0];
            const ptrdiff_t src_stride1 = a.stride()[1];
            const ptrdiff_t dst_stride0 = new_tensor.stride()[0];
            const ptrdiff_t dst_stride1 = new_tensor.stride()[1];
            scalar_t* dst_data = new_tensor.data_ptr<scalar_t>();
            const scalar_t* src_data = a.data_ptr<scalar_t>();

#pragma omp parallel for if (a.numel() >= OMP_THRESHOLD)
            for (ptrdiff_t r = 0; r < static_cast<ptrdiff_t>(rows); ++r) {
#pragma omp simd
                for (ptrdiff_t c = 0; c < static_cast<ptrdiff_t>(cols); ++c) {
                    dst_data[r * dst_stride0 + c * dst_stride1] = src_data[r * src_stride0 + c * src_stride1];
                }
            }
        } else {
            if (a.rank() > 8) {
                throw std::invalid_argument("Operation on non-contiguous tensor with rank > 8 is not supported");
            }
            scalar_t* dst_data = new_tensor.data_ptr<scalar_t>();
            const scalar_t* src_data = a.data_ptr<scalar_t>();
            const size_t total_elements = a.numel();

#pragma omp parallel if (total_elements >= OMP_NON_CONTIGUOUS_THRESHOLD)
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
                    BinaryNDIterator it(a.shape());
                    it.init_from_flat(start);
                    ptrdiff_t offset_src = it.compute_offset(a.stride());
                    ptrdiff_t offset_dst = it.compute_offset(new_tensor.stride());

                    for (size_t i = start; i < end; ++i) {
                        dst_data[offset_dst] = src_data[offset_src];
                        it.advance(offset_src, a.stride(), offset_dst, new_tensor.stride());
                    }
                }
            }
        }
    }

    Tensor Dispatcher::clone(const Tensor& a) {
        Tensor new_tensor(a.shape(), a.dtype(), a.device());

        HELIX_DISPATCH_ALL_TYPES(a.dtype(), "clone", [&] { clone_impl<scalar_t>(a, new_tensor); });

        if (g_graph_builder) {
            std::unordered_map<std::string, std::any> attributes;
            g_graph_builder->build(
                OperationContext{OpCategory::View, OpType::Clone, new_tensor, {a}, std::move(attributes)}
            );
        }
        return new_tensor;
    }

    Tensor Dispatcher::view(const Tensor& a, Shape new_shape) {
        if (new_shape.numel() != a.numel()) {
            throw std::invalid_argument("view shape must have the same number of elements");
        }
        if (!a.is_contiguous()) {
            throw std::runtime_error("view cannot be called on non-contiguous tensor, use reshape instead");
        }
        auto a_impl = a.impl();
        const auto new_impl = std::make_shared<TensorImpl>(
            a_impl->storage(),
            a_impl->storage_offset(),
            new_shape,
            Stride::compute_contiguous(new_shape),
            a.dtype(),
            a.device()
        );
        Tensor out(new_impl);

        if (g_graph_builder) {
            std::unordered_map<std::string, std::any> attributes;
            attributes["original_shape"] = a.shape();
            g_graph_builder->build(OperationContext{OpCategory::View, OpType::View, out, {a}, std::move(attributes)});
        }
        return out;
    }

    Tensor Dispatcher::slice(const Tensor& a, size_t dim, size_t start, size_t end) {
        if (dim >= a.rank()) {
            throw std::out_of_range("slice dimension out of range");
        }
        if (start > end || end > a.shape()[dim]) {
            throw std::invalid_argument("invalid slice bounds");
        }

        std::vector<size_t> new_dims = a.shape().vec();
        new_dims[dim] = end - start;

        auto a_impl = a.impl();
        size_t new_offset = static_cast<size_t>(
            static_cast<ptrdiff_t>(a_impl->storage_offset()) + static_cast<ptrdiff_t>(start) * a.stride()[dim]
        );

        const auto new_impl = std::make_shared<TensorImpl>(
            a_impl->storage(), new_offset, Shape(new_dims), a.stride(), a.dtype(), a.device()
        );
        Tensor out(new_impl);

        if (g_graph_builder) {
            std::unordered_map<std::string, std::any> attributes;
            attributes["dim"] = dim;
            attributes["start"] = start;
            attributes["end"] = end;
            attributes["input_shape"] = a.shape();
            g_graph_builder->build(OperationContext{OpCategory::View, OpType::Slice, out, {a}, std::move(attributes)});
        }
        return out;
    }

    Tensor Dispatcher::transpose(const Tensor& a, size_t dim0, size_t dim1) {
        if (dim0 >= a.rank() || dim1 >= a.rank()) {
            throw std::out_of_range("transpose dimensions out of range");
        }

        std::vector<size_t> new_dims = a.shape().vec();
        std::swap(new_dims[dim0], new_dims[dim1]);

        std::vector<ptrdiff_t> new_strides = a.stride().vec();
        std::swap(new_strides[dim0], new_strides[dim1]);

        auto a_impl = a.impl();
        const auto new_impl = std::make_shared<TensorImpl>(
            a_impl->storage(), a_impl->storage_offset(), Shape(new_dims), Stride(new_strides), a.dtype(), a.device()
        );
        Tensor out(new_impl);

        if (g_graph_builder) {
            std::unordered_map<std::string, std::any> attributes;
            attributes["dim0"] = dim0;
            attributes["dim1"] = dim1;
            g_graph_builder->build(
                OperationContext{OpCategory::View, OpType::Transpose, out, {a}, std::move(attributes)}
            );
        }
        return out;
    }

    Tensor Dispatcher::broadcast_to(const Tensor& a, Shape new_shape) {
        Tensor out = a.broadcast_to_view(new_shape);

        if (g_graph_builder) {
            std::unordered_map<std::string, std::any> attributes;
            attributes["input_shape"] = a.shape();
            g_graph_builder->build(
                OperationContext{OpCategory::View, OpType::BroadcastTo, out, {a}, std::move(attributes)}
            );
        }
        return out;
    }

    Tensor Dispatcher::unsqueeze(const Tensor& a, int64_t dim) {
        const size_t r = a.rank();
        const int64_t max_dim = static_cast<int64_t>(r);
        const int64_t min_dim = -static_cast<int64_t>(r + 1);

        if (dim < min_dim || dim > max_dim) {
            throw std::out_of_range("unsqueeze dimension out of range");
        }

        const size_t norm_dim = static_cast<size_t>(dim < 0 ? dim + static_cast<int64_t>(r + 1) : dim);

        ptrdiff_t inserted_stride = 1;
        if (norm_dim < r) {
            inserted_stride = checked_stride_mul(a.stride()[norm_dim], a.shape()[norm_dim]);
        }

        std::vector<size_t> new_dims;
        new_dims.reserve(r + 1);
        for (size_t i = 0; i < norm_dim; ++i) {
            new_dims.push_back(a.shape()[i]);
        }
        new_dims.push_back(1);
        for (size_t i = norm_dim; i < r; ++i) {
            new_dims.push_back(a.shape()[i]);
        }

        std::vector<ptrdiff_t> new_strides;
        new_strides.reserve(r + 1);
        for (size_t i = 0; i < norm_dim; ++i) {
            new_strides.push_back(a.stride()[i]);
        }
        new_strides.push_back(inserted_stride);
        for (size_t i = norm_dim; i < r; ++i) {
            new_strides.push_back(a.stride()[i]);
        }

        auto a_impl = a.impl();
        const auto new_impl = std::make_shared<TensorImpl>(
            a_impl->storage(),
            a_impl->storage_offset(),
            Shape(std::move(new_dims)),
            Stride(std::move(new_strides)),
            a.dtype(),
            a.device()
        );
        Tensor out(new_impl);

        if (g_graph_builder) {
            std::unordered_map<std::string, std::any> attributes;
            attributes["original_shape"] = a.shape();
            g_graph_builder->build(OperationContext{OpCategory::View, OpType::View, out, {a}, std::move(attributes)});
        }
        return out;
    }

    Tensor Dispatcher::squeeze(const Tensor& a) {
        const size_t r = a.rank();
        if (r == 0) {
            return a;
        }

        bool has_singleton = false;
        for (size_t i = 0; i < r; ++i) {
            if (a.shape()[i] == 1) {
                has_singleton = true;
                break;
            }
        }

        if (!has_singleton) {
            return a;
        }

        std::vector<size_t> new_dims;
        std::vector<ptrdiff_t> new_strides;
        for (size_t i = 0; i < r; ++i) {
            if (a.shape()[i] != 1) {
                new_dims.push_back(a.shape()[i]);
                new_strides.push_back(a.stride()[i]);
            }
        }

        auto a_impl = a.impl();
        const auto new_impl = std::make_shared<TensorImpl>(
            a_impl->storage(),
            a_impl->storage_offset(),
            Shape(std::move(new_dims)),
            Stride(std::move(new_strides)),
            a.dtype(),
            a.device()
        );
        Tensor out(new_impl);

        if (g_graph_builder) {
            std::unordered_map<std::string, std::any> attributes;
            attributes["original_shape"] = a.shape();
            g_graph_builder->build(OperationContext{OpCategory::View, OpType::View, out, {a}, std::move(attributes)});
        }
        return out;
    }

    Tensor Dispatcher::squeeze(const Tensor& a, int64_t dim) {
        const size_t r = a.rank();
        if (r == 0) {
            if (dim == 0 || dim == -1) {
                return a;
            }
            throw std::out_of_range("squeeze dimension out of range");
        }

        const int64_t max_dim = static_cast<int64_t>(r) - 1;
        const int64_t min_dim = -static_cast<int64_t>(r);

        if (dim < min_dim || dim > max_dim) {
            throw std::out_of_range("squeeze dimension out of range");
        }

        const size_t norm_dim = static_cast<size_t>(dim < 0 ? dim + static_cast<int64_t>(r) : dim);

        if (a.shape()[norm_dim] != 1) {
            return a;
        }

        std::vector<size_t> new_dims;
        new_dims.reserve(r - 1);
        std::vector<ptrdiff_t> new_strides;
        new_strides.reserve(r - 1);
        for (size_t i = 0; i < r; ++i) {
            if (i != norm_dim) {
                new_dims.push_back(a.shape()[i]);
                new_strides.push_back(a.stride()[i]);
            }
        }

        auto a_impl = a.impl();
        const auto new_impl = std::make_shared<TensorImpl>(
            a_impl->storage(),
            a_impl->storage_offset(),
            Shape(std::move(new_dims)),
            Stride(std::move(new_strides)),
            a.dtype(),
            a.device()
        );
        Tensor out(new_impl);

        if (g_graph_builder) {
            std::unordered_map<std::string, std::any> attributes;
            attributes["original_shape"] = a.shape();
            g_graph_builder->build(OperationContext{OpCategory::View, OpType::View, out, {a}, std::move(attributes)});
        }
        return out;
    }

    Tensor Dispatcher::cat(const std::vector<Tensor>& tensors, size_t dim) {
        if (tensors.empty()) {
            throw std::invalid_argument("cat expects a non-empty list of tensors");
        }

        const size_t rank = tensors[0].rank();
        if (dim >= rank) {
            throw std::out_of_range("cat dimension out of range");
        }

        const DType dtype = tensors[0].dtype();
        const Device device = tensors[0].device();
        const Shape& first_shape = tensors[0].shape();

        size_t cat_dim_size = 0;
        for (const auto& t : tensors) {
            if (t.rank() != rank) {
                throw std::invalid_argument("cat tensors must all have the same rank");
            }
            if (t.dtype() != dtype) {
                throw std::invalid_argument("cat tensors must have the same dtype");
            }
            if (t.device() != device) {
                throw std::invalid_argument("cat tensors must be on the same device");
            }
            for (size_t d = 0; d < rank; ++d) {
                if (d != dim && t.shape()[d] != first_shape[d]) {
                    throw std::invalid_argument(
                        "cat tensors must have matching shapes along non-concatenated dimensions"
                    );
                }
            }
            if (add_overflow(cat_dim_size, t.shape()[dim], &cat_dim_size)) {
                throw std::overflow_error("cat dimension size exceeds maximum size_t");
            }
        }

        std::vector<size_t> out_dims = first_shape.vec();
        out_dims[dim] = cat_dim_size;
        const Shape out_shape(std::move(out_dims));

        Tensor out(out_shape, dtype, device);

        if (out.numel() > 0) {
            size_t outer_size = 1;
            for (size_t d = 0; d < dim; ++d) outer_size *= out_shape[d];
            size_t inner_size = 1;
            for (size_t d = dim + 1; d < rank; ++d) inner_size *= out_shape[d];

            if (device.is_cpu()) {
                HELIX_DISPATCH_ALL_TYPES(dtype, "cat", [&] {
                    scalar_t* out_data = out.data_ptr<scalar_t>();
                    size_t dim_offset = 0;

                    for (const auto& t : tensors) {
                        const size_t d_k = t.shape()[dim];
                        if (d_k == 0) continue;

                        if (t.is_contiguous()) {
                            const scalar_t* src_data = t.data_ptr<scalar_t>();
                            const size_t chunk_elems = d_k * inner_size;
                            const size_t chunk_bytes = chunk_elems * sizeof(scalar_t);

                            for (size_t i = 0; i < outer_size; ++i) {
                                scalar_t* dst_ptr = out_data + (i * cat_dim_size + dim_offset) * inner_size;
                                const scalar_t* src_ptr = src_data + i * chunk_elems;
                                std::memcpy(dst_ptr, src_ptr, chunk_bytes);
                            }
                        } else {
                            const scalar_t* src_data = t.data_ptr<scalar_t>();
                            const auto& t_shape = t.shape();
                            const auto& t_stride = t.stride();

                            bool inner_contiguous = true;
                            for (size_t d = dim + 1; d < rank; ++d) {
                                const ptrdiff_t expected_stride =
                                    (d + 1 < rank) ? static_cast<ptrdiff_t>(t_stride[d + 1] * t_shape[d + 1]) : 1;
                                if (t_stride[d] != expected_stride) {
                                    inner_contiguous = false;
                                    break;
                                }
                            }

                            if (inner_contiguous && inner_size > 0) {
                                const size_t inner_bytes = inner_size * sizeof(scalar_t);
                                for (size_t i = 0; i < outer_size; ++i) {
                                    ptrdiff_t outer_offset = 0;
                                    size_t rem = i;
                                    for (size_t d = dim; d > 0; --d) {
                                        size_t axis = d - 1;
                                        size_t idx = rem % t_shape[axis];
                                        rem /= t_shape[axis];
                                        outer_offset += static_cast<ptrdiff_t>(idx) * t_stride[axis];
                                    }

                                    for (size_t j = 0; j < d_k; ++j) {
                                        scalar_t* dst_ptr =
                                            out_data + ((i * cat_dim_size + dim_offset + j) * inner_size);
                                        const scalar_t* src_ptr =
                                            src_data + outer_offset + static_cast<ptrdiff_t>(j) * t_stride[dim];
                                        std::memcpy(dst_ptr, src_ptr, inner_bytes);
                                    }
                                }
                            } else {
                                for (size_t i = 0; i < outer_size; ++i) {
                                    ptrdiff_t outer_offset = 0;
                                    size_t rem = i;
                                    for (size_t d = dim; d > 0; --d) {
                                        size_t axis = d - 1;
                                        size_t idx = rem % t_shape[axis];
                                        rem /= t_shape[axis];
                                        outer_offset += static_cast<ptrdiff_t>(idx) * t_stride[axis];
                                    }

                                    for (size_t j = 0; j < d_k; ++j) {
                                        ptrdiff_t dim_src_offset =
                                            outer_offset + static_cast<ptrdiff_t>(j) * t_stride[dim];
                                        scalar_t* dst_ptr =
                                            out_data + ((i * cat_dim_size + dim_offset + j) * inner_size);

                                        for (size_t m = 0; m < inner_size; ++m) {
                                            ptrdiff_t inner_offset = 0;
                                            size_t inner_rem = m;
                                            for (size_t d = rank; d > dim + 1; --d) {
                                                size_t axis = d - 1;
                                                size_t idx = inner_rem % t_shape[axis];
                                                inner_rem /= t_shape[axis];
                                                inner_offset += static_cast<ptrdiff_t>(idx) * t_stride[axis];
                                            }
                                            dst_ptr[m] = src_data[dim_src_offset + inner_offset];
                                        }
                                    }
                                }
                            }
                        }

                        dim_offset += d_k;
                    }
                });
            } else {
                throw std::runtime_error("Unsupported device");
            }
        }

        if (g_graph_builder) {
            std::unordered_map<std::string, std::any> attributes;
            attributes["dim"] = dim;
            std::vector<size_t> split_sizes;
            split_sizes.reserve(tensors.size());
            for (const auto& t : tensors) {
                split_sizes.push_back(t.shape()[dim]);
            }
            attributes["split_sizes"] = std::move(split_sizes);
            std::vector<std::reference_wrapper<const Tensor>> input_refs;
            input_refs.reserve(tensors.size());
            for (const auto& t : tensors) {
                input_refs.push_back(std::cref(t));
            }
            g_graph_builder->build(OperationContext{
                .category = OpCategory::View,
                .type = OpType::Cat,
                .out = out,
                .inputs = std::move(input_refs),
                .attributes = std::move(attributes)
            });
        }

        return out;
    }

    Tensor Dispatcher::ensure_contiguous(const Tensor& t) { return t.contiguous(); }

    Tensor Dispatcher::cast(const Tensor& a, DType new_dtype) {
        if (a.dtype() == new_dtype) return clone(a);
        Tensor out(a.shape(), new_dtype, a.device());
        Tensor lhs = ensure_contiguous(a);

        HELIX_DISPATCH_ALL_TYPES(a.dtype(), "cast_src", [&] { dispatch_cast_dst<scalar_t>(new_dtype, lhs, out); });

        if (g_graph_builder) {
            std::unordered_map<std::string, std::any> attrs;
            attrs["dtype"] = dtype_name(new_dtype);
            g_graph_builder->build(OperationContext{OpCategory::View, OpType::Cast, out, {a}, std::move(attrs)});
        }
        return out;
    }

    Tensor Dispatcher::add(const Tensor& a, const Tensor& b) {
        const Shape out_shape = compute_broadcast_shape(a.shape(), b.shape());
        const DType out_dtype = promote_types(a.dtype(), b.dtype());
        Tensor lhs = (a.dtype() == out_dtype) ? a : cast(a, out_dtype);
        Tensor rhs = (b.dtype() == out_dtype) ? b : cast(b, out_dtype);

        Tensor out(out_shape, out_dtype, a.device());
        if (a.device().is_cpu()) {
            HELIX_DISPATCH_ALL_TYPES(out_dtype, "add", [&] {
                run_binary_kernel<scalar_t>(lhs, rhs, out, CPUBackend::add<scalar_t>, [](scalar_t lhs, scalar_t rhs) {
                    return lhs + rhs;
                });
            });
        } else {
            throw std::runtime_error("Unsupported device");
        }
        if (g_graph_builder) {
            g_graph_builder->build(OperationContext{OpCategory::Binary, OpType::Add, out, {a, b}});
        }
        return out;
    }

    void Dispatcher::add_(Tensor& a, const Tensor& b) {
        if (g_graph_builder && (a.requires_grad() || b.requires_grad())) {
            if (a.requires_grad()) {
                if (a.is_leaf()) {
                    throw std::runtime_error("a leaf Tensor that requires grad is being used in an in-place operation."
                    );
                }
                throw std::runtime_error(
                    "in-place operations on non-leaf tensors that require grad are currently not supported in HELIX "
                    "Autograd."
                );
            }
            throw std::runtime_error(
                "in-place operations involving tensors that require grad are currently not supported in HELIX "
                "Autograd."
            );
        }

        if (a.shape() != b.shape()) {
            throw std::invalid_argument("Inplace addition requires matching shapes without broadcasting.");
        }

        if (a.device() != b.device()) {
            throw std::invalid_argument("Inplace addition requires both tensors to be on the same device.");
        }

        if (a.has_internal_overlap()) {
            throw std::runtime_error("add_: in-place operation on a tensor with overlapping memory is not supported.");
        }

        if (a.dtype() != b.dtype()) {
            throw std::invalid_argument("add_: inplace addition requires both tensors to have the same dtype.");
        }

        const bool is_aliased = (a.impl()->storage() == b.impl()->storage()) &&
                                (a.data_ptr() != b.data_ptr() || a.stride() != b.stride() || a.shape() != b.shape());
        Tensor safe_b = is_aliased ? b.clone() : b;

        if (a.device().is_cpu()) {
            HELIX_DISPATCH_ALL_TYPES(a.dtype(), "add_", [&] { add_inplace_kernel<scalar_t>(a, safe_b); });
        } else {
            throw std::runtime_error("Unsupported device");
        }
    }

    Tensor Dispatcher::sub(const Tensor& a, const Tensor& b) {
        const Shape out_shape = compute_broadcast_shape(a.shape(), b.shape());
        const DType out_dtype = promote_types(a.dtype(), b.dtype());
        Tensor lhs = (a.dtype() == out_dtype) ? a : cast(a, out_dtype);
        Tensor rhs = (b.dtype() == out_dtype) ? b : cast(b, out_dtype);

        Tensor out(out_shape, out_dtype, a.device());
        if (a.device().is_cpu()) {
            HELIX_DISPATCH_ALL_TYPES(out_dtype, "sub", [&] {
                run_binary_kernel<scalar_t>(lhs, rhs, out, CPUBackend::sub<scalar_t>, [](scalar_t lhs, scalar_t rhs) {
                    return lhs - rhs;
                });
            });
        } else {
            throw std::runtime_error("Unsupported device");
        }
        if (g_graph_builder) {
            g_graph_builder->build(
                OperationContext{.category = OpCategory::Binary, .type = OpType::Sub, .out = out, .inputs = {a, b}}
            );
        }
        return out;
    }

    Tensor Dispatcher::mul(const Tensor& a, const Tensor& b) {
        const Shape out_shape = compute_broadcast_shape(a.shape(), b.shape());
        const DType out_dtype = promote_types(a.dtype(), b.dtype());
        Tensor lhs = (a.dtype() == out_dtype) ? a : cast(a, out_dtype);
        Tensor rhs = (b.dtype() == out_dtype) ? b : cast(b, out_dtype);

        Tensor out(out_shape, out_dtype, a.device());
        if (a.device().is_cpu()) {
            HELIX_DISPATCH_ALL_TYPES(out_dtype, "mul", [&] {
                run_binary_kernel<scalar_t>(lhs, rhs, out, CPUBackend::mul<scalar_t>, [](scalar_t lhs, scalar_t rhs) {
                    return lhs * rhs;
                });
            });
        } else {
            throw std::runtime_error("Unsupported device");
        }
        if (g_graph_builder) {
            g_graph_builder->build(
                OperationContext{.category = OpCategory::Binary, .type = OpType::Mul, .out = out, .inputs = {a, b}}
            );
        }
        return out;
    }

    Tensor Dispatcher::div(const Tensor& a, const Tensor& b) {
        const Shape out_shape = compute_broadcast_shape(a.shape(), b.shape());
        const DType out_dtype = promote_to_float(promote_types(a.dtype(), b.dtype()));
        Tensor lhs = (a.dtype() == out_dtype) ? a : cast(a, out_dtype);
        Tensor rhs = (b.dtype() == out_dtype) ? b : cast(b, out_dtype);

        Tensor out(out_shape, out_dtype, a.device());
        if (a.device().is_cpu()) {
            HELIX_DISPATCH_ALL_TYPES(out_dtype, "div", [&] {
                run_binary_kernel<scalar_t>(lhs, rhs, out, CPUBackend::div<scalar_t>, [](scalar_t lhs, scalar_t rhs) {
                    return lhs / rhs;
                });
            });
        } else {
            throw std::runtime_error("Unsupported device");
        }
        if (g_graph_builder) {
            g_graph_builder->build(
                OperationContext{.category = OpCategory::Binary, .type = OpType::Div, .out = out, .inputs = {a, b}}
            );
        }
        return out;
    }

    Tensor Dispatcher::add_scalar(const Tensor& a, const float scalar) {
        const DType out_dtype = promote_types(a.dtype(), DType::Float32);
        Tensor lhs = ensure_contiguous((a.dtype() == out_dtype) ? a : cast(a, out_dtype));
        Tensor out(a.shape(), out_dtype, a.device());
        if (a.device().is_cpu()) {
            HELIX_DISPATCH_ALL_TYPES(out_dtype, "add_scalar", [&] {
                CPUBackend::add_scalar(
                    lhs.data_ptr<scalar_t>(), static_cast<scalar_t>(scalar), out.data_ptr<scalar_t>(), out.numel()
                );
            });
        } else {
            throw std::runtime_error("Unsupported device");
        }
        if (g_graph_builder) {
            std::unordered_map<std::string, std::any> attrs;
            attrs["scalar"] = scalar;
            g_graph_builder->build(OperationContext{
                .category = OpCategory::Unary,
                .type = OpType::AddScalar,
                .out = out,
                .inputs = {a},
                .attributes = std::move(attrs)
            });
        }
        return out;
    }

    Tensor Dispatcher::sub_scalar(const Tensor& a, const float scalar) {
        const DType out_dtype = promote_types(a.dtype(), DType::Float32);
        Tensor lhs = ensure_contiguous((a.dtype() == out_dtype) ? a : cast(a, out_dtype));
        Tensor out(a.shape(), out_dtype, a.device());
        if (a.device().is_cpu()) {
            HELIX_DISPATCH_ALL_TYPES(out_dtype, "sub_scalar", [&] {
                CPUBackend::sub_scalar(
                    lhs.data_ptr<scalar_t>(), static_cast<scalar_t>(scalar), out.data_ptr<scalar_t>(), out.numel()
                );
            });
        } else {
            throw std::runtime_error("Unsupported device");
        }
        if (g_graph_builder) {
            std::unordered_map<std::string, std::any> attrs;
            attrs["scalar"] = scalar;
            g_graph_builder->build(OperationContext{
                .category = OpCategory::Unary,
                .type = OpType::SubScalar,
                .out = out,
                .inputs = {a},
                .attributes = std::move(attrs)
            });
        }
        return out;
    }

    Tensor Dispatcher::mul_scalar(const Tensor& a, const float scalar) {
        const DType out_dtype = promote_types(a.dtype(), DType::Float32);
        Tensor lhs = ensure_contiguous((a.dtype() == out_dtype) ? a : cast(a, out_dtype));
        Tensor out(a.shape(), out_dtype, a.device());
        if (a.device().is_cpu()) {
            HELIX_DISPATCH_ALL_TYPES(out_dtype, "mul_scalar", [&] {
                CPUBackend::mul_scalar(
                    lhs.data_ptr<scalar_t>(), static_cast<scalar_t>(scalar), out.data_ptr<scalar_t>(), out.numel()
                );
            });
        } else {
            throw std::runtime_error("Unsupported device");
        }
        if (g_graph_builder) {
            std::unordered_map<std::string, std::any> attrs;
            attrs["scalar"] = scalar;
            g_graph_builder->build(OperationContext{
                .category = OpCategory::Unary,
                .type = OpType::MulScalar,
                .out = out,
                .inputs = {a},
                .attributes = std::move(attrs)
            });
        }
        return out;
    }

    Tensor Dispatcher::div_scalar(const Tensor& a, const float scalar) {
        const DType out_dtype = promote_types(a.dtype(), DType::Float32);
        Tensor lhs = ensure_contiguous((a.dtype() == out_dtype) ? a : cast(a, out_dtype));
        Tensor out(a.shape(), out_dtype, a.device());
        if (a.device().is_cpu()) {
            HELIX_DISPATCH_ALL_TYPES(out_dtype, "div_scalar", [&] {
                CPUBackend::div_scalar(
                    lhs.data_ptr<scalar_t>(), static_cast<scalar_t>(scalar), out.data_ptr<scalar_t>(), out.numel()
                );
            });
        } else {
            throw std::runtime_error("Unsupported device");
        }
        if (g_graph_builder) {
            std::unordered_map<std::string, std::any> attrs;
            attrs["scalar"] = scalar;
            g_graph_builder->build(OperationContext{
                .category = OpCategory::Unary,
                .type = OpType::DivScalar,
                .out = out,
                .inputs = {a},
                .attributes = std::move(attrs)
            });
        }
        return out;
    }

    Tensor Dispatcher::neg(const Tensor& a) {
        Tensor lhs = ensure_contiguous(a);
        Tensor out(a.shape(), a.dtype(), a.device());
        if (a.device().is_cpu()) {
            HELIX_DISPATCH_ALL_TYPES(a.dtype(), "neg", [&] {
                CPUBackend::neg(lhs.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), out.numel());
            });
        } else {
            throw std::runtime_error("Unsupported device");
        }
        if (g_graph_builder) {
            g_graph_builder->build(
                OperationContext{.category = OpCategory::Unary, .type = OpType::Neg, .out = out, .inputs = {a}}
            );
        }
        return out;
    }

    Tensor Dispatcher::exp(const Tensor& a) {
        const DType out_dtype = promote_to_float(a.dtype());
        Tensor lhs = ensure_contiguous((a.dtype() == out_dtype) ? a : cast(a, out_dtype));
        Tensor out(a.shape(), out_dtype, a.device());
        if (a.device().is_cpu()) {
            HELIX_DISPATCH_ALL_TYPES(out_dtype, "exp", [&] {
                CPUBackend::exp(lhs.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), out.numel());
            });
        } else {
            throw std::runtime_error("Unsupported device");
        }
        if (g_graph_builder) {
            g_graph_builder->build(
                OperationContext{.category = OpCategory::Unary, .type = OpType::Exp, .out = out, .inputs = {a}}
            );
        }
        return out;
    }

    Tensor Dispatcher::tanh(const Tensor& a) {
        const DType out_dtype = promote_to_float(a.dtype());
        Tensor lhs = ensure_contiguous((a.dtype() == out_dtype) ? a : cast(a, out_dtype));
        Tensor out(a.shape(), out_dtype, a.device());
        if (a.device().is_cpu()) {
            HELIX_DISPATCH_ALL_TYPES(out_dtype, "tanh", [&] {
                CPUBackend::tanh(lhs.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), out.numel());
            });
        } else {
            throw std::runtime_error("Unsupported device");
        }
        if (g_graph_builder) {
            g_graph_builder->build(
                OperationContext{.category = OpCategory::Unary, .type = OpType::Tanh, .out = out, .inputs = {a}}
            );
        }
        return out;
    }

    Tensor Dispatcher::log(const Tensor& a) {
        const DType out_dtype = promote_to_float(a.dtype());
        Tensor lhs = ensure_contiguous((a.dtype() == out_dtype) ? a : cast(a, out_dtype));
        Tensor out(a.shape(), out_dtype, a.device());
        if (a.device().is_cpu()) {
            HELIX_DISPATCH_ALL_TYPES(out_dtype, "log", [&] {
                CPUBackend::log(lhs.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), out.numel());
            });
        } else {
            throw std::runtime_error("Unsupported device");
        }
        if (g_graph_builder) {
            g_graph_builder->build(
                OperationContext{.category = OpCategory::Unary, .type = OpType::Log, .out = out, .inputs = {a}}
            );
        }
        return out;
    }

    Tensor Dispatcher::sqrt(const Tensor& a) {
        const DType out_dtype = promote_to_float(a.dtype());
        Tensor lhs = ensure_contiguous((a.dtype() == out_dtype) ? a : cast(a, out_dtype));
        Tensor out(a.shape(), out_dtype, a.device());
        if (a.device().is_cpu()) {
            HELIX_DISPATCH_ALL_TYPES(out_dtype, "sqrt", [&] {
                CPUBackend::sqrt(lhs.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), out.numel());
            });
        } else {
            throw std::runtime_error("Unsupported device");
        }
        if (g_graph_builder) {
            g_graph_builder->build(
                OperationContext{.category = OpCategory::Unary, .type = OpType::Sqrt, .out = out, .inputs = {a}}
            );
        }
        return out;
    }

    Tensor Dispatcher::relu(const Tensor& a) {
        Tensor lhs = ensure_contiguous(a);
        Tensor out(a.shape(), a.dtype(), a.device());
        if (a.device().is_cpu()) {
            HELIX_DISPATCH_ALL_TYPES(a.dtype(), "relu", [&] {
                CPUBackend::relu(lhs.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), out.numel());
            });
        } else {
            throw std::runtime_error("Unsupported device");
        }
        if (g_graph_builder) {
            g_graph_builder->build(
                OperationContext{.category = OpCategory::Unary, .type = OpType::ReLU, .out = out, .inputs = {a}}
            );
        }
        return out;
    }

    Tensor Dispatcher::relu_backward(const Tensor& grad_out, const Tensor& a) {
        if (grad_out.shape() != a.shape()) {
            throw std::invalid_argument("Shape mismatch in relu_backward");
        }
        if (grad_out.device() != a.device()) {
            throw std::invalid_argument("Device mismatch in relu_backward");
        }

        const DType target_dtype = grad_out.dtype();
        Tensor lhs = ensure_contiguous(grad_out);
        Tensor rhs = ensure_contiguous((a.dtype() == target_dtype) ? a : cast(a, target_dtype));
        Tensor out(grad_out.shape(), target_dtype, grad_out.device());

        if (grad_out.device().is_cpu()) {
            HELIX_DISPATCH_ALL_TYPES(target_dtype, "relu_backward", [&] {
                CPUBackend::relu_backward(
                    lhs.data_ptr<scalar_t>(), rhs.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), out.numel()
                );
            });
        } else {
            throw std::runtime_error("Unsupported device");
        }

        return out;
    }

    Tensor Dispatcher::pow(const Tensor& a, float exponent) {
        const DType out_dtype = promote_to_float(a.dtype());
        Tensor lhs = ensure_contiguous((a.dtype() == out_dtype) ? a : cast(a, out_dtype));
        Tensor out(a.shape(), out_dtype, a.device());
        if (a.device().is_cpu()) {
            HELIX_DISPATCH_ALL_TYPES(out_dtype, "pow", [&] {
                CPUBackend::pow(
                    lhs.data_ptr<scalar_t>(), static_cast<scalar_t>(exponent), out.data_ptr<scalar_t>(), out.numel()
                );
            });
        } else {
            throw std::runtime_error("Unsupported device");
        }
        if (g_graph_builder) {
            OperationContext ctx{.category = OpCategory::Unary, .type = OpType::Pow, .out = out, .inputs = {a}};
            ctx.attributes["exponent"] = exponent;
            g_graph_builder->build(ctx);
        }
        return out;
    }

    Tensor Dispatcher::matmul(const Tensor& a, const Tensor& b) {
        if (a.rank() != 2 || b.rank() != 2) {
            throw std::invalid_argument("matmul currently only supports 2D tensors");
        }
        if (a.shape()[1] != b.shape()[0]) {
            throw std::invalid_argument("matmul shapes incompatible");
        }

        const DType out_dtype = promote_types(a.dtype(), b.dtype());
        Tensor lhs = ensure_contiguous(a);
        Tensor rhs = ensure_contiguous(b);
        if (lhs.dtype() != out_dtype) lhs = cast(lhs, out_dtype);
        if (rhs.dtype() != out_dtype) rhs = cast(rhs, out_dtype);

        const size_t M = a.shape()[0];
        const size_t K = a.shape()[1];
        const size_t N = b.shape()[1];

        Tensor out(Shape{M, N}, out_dtype, a.device());

        if (a.device().is_cpu()) {
            HELIX_DISPATCH_ALL_TYPES(out_dtype, "matmul", [&] {
                CPUBackend::matmul(
                    lhs.data_ptr<scalar_t>(), rhs.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), M, K, N
                );
            });
        } else {
            throw std::runtime_error("Unsupported device");
        }
        if (g_graph_builder) {
            g_graph_builder->build(
                OperationContext{.category = OpCategory::Matrix, .type = OpType::MatMul, .out = out, .inputs = {a, b}}
            );
        }
        return out;
    }

    Tensor Dispatcher::bmm(const Tensor& a, const Tensor& b) {
        if (a.rank() != 3 || b.rank() != 3) {
            throw std::invalid_argument(
                "bmm expects 3D tensors, but got rank " + std::to_string(a.rank()) + " and " + std::to_string(b.rank())
            );
        }
        if (a.shape()[0] != b.shape()[0]) {
            throw std::invalid_argument(
                "bmm batch dimensions must match: " + std::to_string(a.shape()[0]) + " vs " +
                std::to_string(b.shape()[0])
            );
        }
        if (a.shape()[2] != b.shape()[1]) {
            throw std::invalid_argument(
                "bmm inner matrix dimensions must match: " + std::to_string(a.shape()[2]) + " vs " +
                std::to_string(b.shape()[1])
            );
        }
        if (a.device() != b.device()) {
            throw std::invalid_argument("bmm tensors must be on the same device");
        }
        if (!a.device().is_cpu()) {
            throw std::runtime_error("Unsupported device");
        }

        const DType out_dtype = promote_types(a.dtype(), b.dtype());
        Tensor lhs = (a.dtype() == out_dtype) ? ensure_contiguous(a) : cast(a, out_dtype);
        Tensor rhs = (b.dtype() == out_dtype) ? ensure_contiguous(b) : cast(b, out_dtype);

        const size_t B = a.shape()[0];
        const size_t M = a.shape()[1];
        const size_t K = a.shape()[2];
        const size_t N = b.shape()[2];

        Tensor out(Shape{B, M, N}, out_dtype, a.device());

        if (out.numel() == 0) {
            // Zero-extent tensor: numel is 0, nothing to execute in backend
        } else {
            HELIX_DISPATCH_ALL_TYPES(out_dtype, "bmm", [&] {
                CPUBackend::bmm(
                    lhs.data_ptr<scalar_t>(), rhs.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), B, M, K, N
                );
            });
        }

        if (g_graph_builder) {
            g_graph_builder->build(
                OperationContext{.category = OpCategory::Matrix, .type = OpType::Bmm, .out = out, .inputs = {a, b}}
            );
        }
        return out;
    }

    Tensor Dispatcher::sum(const Tensor& a, std::optional<size_t> axis, bool keepdim) {
        Tensor lhs = ensure_contiguous(a);

        if (!axis.has_value()) {
            const Shape out_shape = keepdim ? Shape(std::vector<size_t>(a.rank(), 1)) : Shape();
            Tensor out(out_shape, a.dtype(), a.device());
            if (a.device().is_cpu()) {
                HELIX_DISPATCH_ALL_TYPES(a.dtype(), "sum", [&] {
                    CPUBackend::sum(lhs.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), 1, a.numel(), 1);
                });
            } else {
                throw std::runtime_error("Unsupported device");
            }
            if (g_graph_builder) {
                OperationContext ctx{.category = OpCategory::Reduce, .type = OpType::Sum, .out = out, .inputs = {a}};
                ctx.attributes["axis"] = axis;
                ctx.attributes["keepdim"] = keepdim;
                g_graph_builder->build(ctx);
            }
            return out;
        }

        const size_t dim = axis.value();
        if (dim >= a.rank()) throw std::out_of_range("axis out of bounds");

        std::vector<size_t> out_dims;
        for (size_t i = 0; i < a.rank(); ++i) {
            if (i == dim) {
                if (keepdim) out_dims.push_back(1);
            } else {
                out_dims.push_back(a.shape()[i]);
            }
        }
        const Shape out_shape(out_dims);
        Tensor out(out_shape, a.dtype(), a.device());

        size_t outer_size = 1;
        for (size_t i = 0; i < dim; ++i) outer_size *= a.shape()[i];
        const size_t dim_size = a.shape()[dim];
        size_t inner_size = 1;
        for (size_t i = dim + 1; i < a.rank(); ++i) inner_size *= a.shape()[i];

        if (a.device().is_cpu()) {
            HELIX_DISPATCH_ALL_TYPES(a.dtype(), "sum", [&] {
                CPUBackend::sum(lhs.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), outer_size, dim_size, inner_size);
            });
        } else {
            throw std::runtime_error("Unsupported device");
        }
        if (g_graph_builder) {
            OperationContext ctx{.category = OpCategory::Reduce, .type = OpType::Sum, .out = out, .inputs = {a}};
            ctx.attributes["axis"] = axis;
            ctx.attributes["keepdim"] = keepdim;
            g_graph_builder->build(ctx);
        }
        return out;
    }

    Tensor Dispatcher::mean(const Tensor& a, std::optional<size_t> axis, bool keepdim) {
        const DType out_dtype = promote_to_float(a.dtype());
        Tensor lhs = ensure_contiguous((a.dtype() == out_dtype) ? a : cast(a, out_dtype));

        if (!axis.has_value()) {
            const Shape out_shape = keepdim ? Shape(std::vector<size_t>(a.rank(), 1)) : Shape();
            Tensor out(out_shape, out_dtype, a.device());
            if (a.device().is_cpu()) {
                HELIX_DISPATCH_ALL_TYPES(out_dtype, "mean", [&] {
                    CPUBackend::mean(lhs.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), 1, a.numel(), 1);
                });
            } else {
                throw std::runtime_error("Unsupported device");
            }
            if (g_graph_builder) {
                OperationContext ctx{.category = OpCategory::Reduce, .type = OpType::Mean, .out = out, .inputs = {a}};
                ctx.attributes["axis"] = axis;
                ctx.attributes["keepdim"] = keepdim;
                g_graph_builder->build(ctx);
            }
            return out;
        }

        const size_t dim = axis.value();
        if (dim >= a.rank()) throw std::out_of_range("axis out of bounds");

        std::vector<size_t> out_dims;
        for (size_t i = 0; i < a.rank(); ++i) {
            if (i == dim) {
                if (keepdim) out_dims.push_back(1);
            } else {
                out_dims.push_back(a.shape()[i]);
            }
        }
        const Shape out_shape(out_dims);
        Tensor out(out_shape, out_dtype, a.device());

        size_t outer_size = 1;
        for (size_t i = 0; i < dim; ++i) outer_size *= a.shape()[i];
        const size_t dim_size = a.shape()[dim];
        size_t inner_size = 1;
        for (size_t i = dim + 1; i < a.rank(); ++i) inner_size *= a.shape()[i];

        if (a.device().is_cpu()) {
            HELIX_DISPATCH_ALL_TYPES(out_dtype, "mean", [&] {
                CPUBackend::mean(lhs.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), outer_size, dim_size, inner_size);
            });
        } else {
            throw std::runtime_error("Unsupported device");
        }
        if (g_graph_builder) {
            OperationContext ctx{.category = OpCategory::Reduce, .type = OpType::Mean, .out = out, .inputs = {a}};
            ctx.attributes["axis"] = axis;
            ctx.attributes["keepdim"] = keepdim;
            g_graph_builder->build(ctx);
        }
        return out;
    }

    Tensor Dispatcher::argmax(const Tensor& a, size_t dim) {
        if (dim >= a.rank()) {
            throw std::out_of_range("argmax dimension out of range");
        }
        if (a.shape()[dim] == 0) {
            throw std::invalid_argument("argmax cannot be performed on an empty dimension");
        }

        Tensor lhs = ensure_contiguous(a);

        Shape out_shape;
        if (a.rank() > 1) {
            std::vector<size_t> out_dims;
            out_dims.reserve(a.rank() - 1);
            for (size_t i = 0; i < a.rank(); ++i) {
                if (i != dim) {
                    out_dims.push_back(a.shape()[i]);
                }
            }
            out_shape = Shape(std::move(out_dims));
        }

        Tensor out(out_shape, DType::Int64, a.device());

        size_t outer_size = 1;
        for (size_t i = 0; i < dim; ++i) outer_size *= a.shape()[i];
        const size_t dim_size = a.shape()[dim];
        size_t inner_size = 1;
        for (size_t i = dim + 1; i < a.rank(); ++i) inner_size *= a.shape()[i];

        if (a.device().is_cpu()) {
            HELIX_DISPATCH_ALL_TYPES(a.dtype(), "argmax", [&] {
                CPUBackend::argmax(lhs.data_ptr<scalar_t>(), out.data_ptr<int64_t>(), outer_size, dim_size, inner_size);
            });
        } else {
            throw std::runtime_error("Unsupported device");
        }

        // Note: argmax is non-differentiable; autograd graph is not constructed.
        return out;
    }

    Tensor Dispatcher::cross_entropy(const Tensor& pred, const Tensor& target) {
        if (pred.rank() != 2 || target.rank() != 2) {
            throw std::invalid_argument("cross_entropy currently only supports 2D tensors");
        }
        if (pred.shape() != target.shape()) {
            throw std::invalid_argument("cross_entropy shapes incompatible");
        }
        if (pred.shape()[0] == 0 || pred.shape()[1] == 0) {
            throw std::invalid_argument("cross_entropy batch size and class count must be non-zero");
        }

        const DType out_dtype = promote_to_float(promote_types(pred.dtype(), target.dtype()));
        Tensor p_contig = ensure_contiguous(pred);
        Tensor t_contig = ensure_contiguous(target);
        if (p_contig.dtype() != out_dtype) p_contig = cast(p_contig, out_dtype);
        if (t_contig.dtype() != out_dtype) t_contig = cast(t_contig, out_dtype);

        const size_t N = pred.shape()[0];
        const size_t C = pred.shape()[1];

        Tensor out(Shape{}, out_dtype, pred.device());                   // scalar loss
        Tensor log_softmax_out(pred.shape(), out_dtype, pred.device());  // [N, C]

        if (pred.device().is_cpu()) {
            HELIX_DISPATCH_ALL_TYPES(out_dtype, "cross_entropy", [&] {
                CPUBackend::cross_entropy(
                    p_contig.data_ptr<scalar_t>(),
                    t_contig.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(),
                    log_softmax_out.data_ptr<scalar_t>(),
                    N,
                    C
                );
            });
        } else {
            throw std::runtime_error("Unsupported device");
        }

        if (g_graph_builder) {
            OperationContext ctx{
                .category = OpCategory::Loss, .type = OpType::CrossEntropy, .out = out, .inputs = {pred, target}
            };
            ctx.attributes["log_softmax"] = log_softmax_out;
            g_graph_builder->build(ctx);
        }

        return out;
    }

    void Dispatcher::sgd(Tensor& param, const Tensor& grad, const float lr) {
        if (param.shape() != grad.shape()) {
            throw std::invalid_argument("SGD requires matching shapes for param and grad.");
        }

        if (param.device() != grad.device()) {
            throw std::invalid_argument("SGD requires parameter and gradient to be on the same device.");
        }

        if (param.dtype() != grad.dtype()) {
            throw std::invalid_argument("SGD requires param and grad to have the same dtype.");
        }

        if (param.has_internal_overlap()) {
            throw std::runtime_error("sgd: in-place operation on a tensor with overlapping memory is not supported.");
        }

        const bool is_aliased =
            (param.impl()->storage() == grad.impl()->storage()) &&
            (param.data_ptr() != grad.data_ptr() || param.stride() != grad.stride() || param.shape() != grad.shape());
        Tensor safe_grad = is_aliased ? grad.clone() : grad;

        if (param.device().is_cpu()) {
            HELIX_DISPATCH_ALL_TYPES(param.dtype(), "sgd", [&] { sgd_inplace_kernel<scalar_t>(param, safe_grad, lr); });
        } else {
            throw std::runtime_error("Unsupported device");
        }

        param.increment_version();
    }

}  // namespace helix
