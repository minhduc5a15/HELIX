#include "core/dispatcher.hpp"

#include <cstring>

#if defined(_OPENMP)
#include <omp.h>
#endif
#include <algorithm>
#include <stdexcept>

#include "backend/cpu_backend.hpp"
#include "core/broadcast.hpp"
#include "core/graph_builder.hpp"
#include "core/nd_iterator.hpp"
#include "core/tensor.hpp"

namespace helix {
    namespace {
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
                const size_t a_stride0 = a.stride()[0];
                const size_t a_stride1 = a.stride()[1];
                const size_t b_stride0 = safe_b.stride()[0];
                const size_t b_stride1 = safe_b.stride()[1];
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
                const size_t p_stride0 = param.stride()[0];
                const size_t p_stride1 = param.stride()[1];
                const size_t g_stride0 = safe_grad.stride()[0];
                const size_t g_stride1 = safe_grad.stride()[1];
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

    static GraphBuilderInterface* g_graph_builder = nullptr;

    void Dispatcher::register_graph_builder(GraphBuilderInterface* builder) { g_graph_builder = builder; }

    GraphBuilderInterface* Dispatcher::get_graph_builder() { return g_graph_builder; }

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
        if (start >= end || end > a.shape()[dim]) {
            throw std::invalid_argument("invalid slice bounds");
        }

        std::vector<size_t> new_dims = a.shape().vec();
        new_dims[dim] = end - start;

        auto a_impl = a.impl();
        size_t new_offset = a_impl->storage_offset() + start * a.stride()[dim];

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

    // NOTE:
    // Current CPU backend only supports contiguous tensors.
    // Once TensorIterator is implemented,
    // remove these contiguous() calls.
    Tensor Dispatcher::add(const Tensor& a, const Tensor& b) {
        const Shape out_shape = compute_broadcast_shape(a.shape(), b.shape());
        const DType out_dtype = promote_types(a.dtype(), b.dtype());
        Tensor lhs = ensure_contiguous(a.broadcast_to_view(out_shape));
        Tensor rhs = ensure_contiguous(b.broadcast_to_view(out_shape));

        if (lhs.dtype() != out_dtype) lhs = cast(lhs, out_dtype);
        if (rhs.dtype() != out_dtype) rhs = cast(rhs, out_dtype);

        Tensor out(out_shape, out_dtype, a.device());
        if (a.device().is_cpu()) {
            HELIX_DISPATCH_ALL_TYPES(out_dtype, "add", [&] {
                CPUBackend::add(
                    lhs.data_ptr<scalar_t>(), rhs.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), out.numel()
                );
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
        Tensor lhs = ensure_contiguous(a.broadcast_to_view(out_shape));
        Tensor rhs = ensure_contiguous(b.broadcast_to_view(out_shape));
        if (lhs.dtype() != out_dtype) lhs = cast(lhs, out_dtype);
        if (rhs.dtype() != out_dtype) rhs = cast(rhs, out_dtype);

        Tensor out(out_shape, out_dtype, a.device());
        if (a.device().is_cpu()) {
            HELIX_DISPATCH_ALL_TYPES(out_dtype, "sub", [&] {
                CPUBackend::sub(
                    lhs.data_ptr<scalar_t>(), rhs.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), out.numel()
                );
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
        Tensor lhs = ensure_contiguous(a.broadcast_to_view(out_shape));
        Tensor rhs = ensure_contiguous(b.broadcast_to_view(out_shape));
        if (lhs.dtype() != out_dtype) lhs = cast(lhs, out_dtype);
        if (rhs.dtype() != out_dtype) rhs = cast(rhs, out_dtype);

        Tensor out(out_shape, out_dtype, a.device());
        if (a.device().is_cpu()) {
            HELIX_DISPATCH_ALL_TYPES(out_dtype, "mul", [&] {
                CPUBackend::mul(
                    lhs.data_ptr<scalar_t>(), rhs.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), out.numel()
                );
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
        Tensor lhs = ensure_contiguous(a.broadcast_to_view(out_shape));
        Tensor rhs = ensure_contiguous(b.broadcast_to_view(out_shape));
        if (lhs.dtype() != out_dtype) lhs = cast(lhs, out_dtype);
        if (rhs.dtype() != out_dtype) rhs = cast(rhs, out_dtype);

        Tensor out(out_shape, out_dtype, a.device());
        if (a.device().is_cpu()) {
            HELIX_DISPATCH_ALL_TYPES(out_dtype, "div", [&] {
                CPUBackend::div(
                    lhs.data_ptr<scalar_t>(), rhs.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), out.numel()
                );
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
        Tensor lhs = ensure_contiguous(grad_out);
        Tensor rhs = ensure_contiguous(a);
        Tensor out(grad_out.shape(), grad_out.dtype(), grad_out.device());
        if (grad_out.device().is_cpu()) {
            HELIX_DISPATCH_ALL_TYPES(grad_out.dtype(), "relu_backward", [&] {
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

    Tensor Dispatcher::cross_entropy(const Tensor& pred, const Tensor& target) {
        if (pred.rank() != 2 || target.rank() != 2) {
            throw std::invalid_argument("cross_entropy currently only supports 2D tensors");
        }
        if (pred.shape() != target.shape()) {
            throw std::invalid_argument("cross_entropy shapes incompatible");
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
