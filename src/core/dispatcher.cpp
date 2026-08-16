#include "core/dispatcher.hpp"

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

    static GraphBuilderInterface* g_graph_builder = nullptr;

    void Dispatcher::register_graph_builder(GraphBuilderInterface* builder) { g_graph_builder = builder; }

    GraphBuilderInterface* Dispatcher::get_graph_builder() { return g_graph_builder; }

    Tensor Dispatcher::ensure_contiguous(const Tensor& t) { return t.contiguous(); }

    // NOTE:
    // Current CPU backend only supports contiguous tensors.
    // Once TensorIterator is implemented,
    // remove these contiguous() calls.
    Tensor Dispatcher::add(const Tensor& a, const Tensor& b) {
        const Shape out_shape = compute_broadcast_shape(a.shape(), b.shape());
        Tensor lhs = ensure_contiguous(a.broadcast_to(out_shape));
        Tensor rhs = ensure_contiguous(b.broadcast_to(out_shape));
        Tensor out(out_shape, a.dtype(), a.device());
        if (a.device().is_cpu())
            CPUBackend::add(lhs.data_ptr(), rhs.data_ptr(), out.data_ptr(), out.numel());
        else
            throw std::runtime_error("Unsupported device");
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

        const bool is_aliased = (a.impl()->storage() == b.impl()->storage()) &&
                                (a.data_ptr() != b.data_ptr() || a.stride() != b.stride() || a.shape() != b.shape());
        Tensor safe_b = is_aliased ? b.clone() : b;

        if (a.device().is_cpu()) {
            if (a.is_contiguous() && safe_b.is_contiguous()) {
                CPUBackend::add(a.data_ptr(), safe_b.data_ptr(), a.data_ptr(), a.numel());
            } else if (a.rank() == 2) {
                const size_t rows = a.shape()[0];
                const size_t cols = a.shape()[1];
                const size_t a_stride0 = a.stride()[0];
                const size_t a_stride1 = a.stride()[1];
                const size_t b_stride0 = safe_b.stride()[0];
                const size_t b_stride1 = safe_b.stride()[1];
                float* a_data = a.data_ptr();
                const float* b_data = safe_b.data_ptr();

#pragma omp parallel for
                for (ptrdiff_t r = 0; r < static_cast<ptrdiff_t>(rows); ++r) {
#pragma omp simd
                    for (size_t c = 0; c < cols; ++c) {
                        a_data[r * a_stride0 + c * a_stride1] += b_data[r * b_stride0 + c * b_stride1];
                    }
                }
            } else {
                float* a_data = a.data_ptr();
                const float* b_data = safe_b.data_ptr();
                const size_t total_elements = a.numel();

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
                        BinaryNDIterator it(a.shape());
                        it.init_from_flat(start);
                        size_t offset_a = it.compute_offset(a.stride());
                        size_t offset_b = it.compute_offset(safe_b.stride());

                        for (size_t i = start; i < end; ++i) {
                            a_data[offset_a] += b_data[offset_b];
                            it.advance(offset_a, a.stride(), offset_b, safe_b.stride());
                        }
                    }
                }
            }
        } else {
            throw std::runtime_error("Unsupported device");
        }
    }

    Tensor Dispatcher::sub(const Tensor& a, const Tensor& b) {
        const Shape out_shape = compute_broadcast_shape(a.shape(), b.shape());
        Tensor lhs = ensure_contiguous(a.broadcast_to(out_shape));
        Tensor rhs = ensure_contiguous(b.broadcast_to(out_shape));
        Tensor out(out_shape, a.dtype(), a.device());
        if (a.device().is_cpu())
            CPUBackend::sub(lhs.data_ptr(), rhs.data_ptr(), out.data_ptr(), out.numel());
        else
            throw std::runtime_error("Unsupported device");
        if (g_graph_builder) {
            g_graph_builder->build(
                OperationContext{.category = OpCategory::Binary, .type = OpType::Sub, .out = out, .inputs = {a, b}}
            );
        }
        return out;
    }

    Tensor Dispatcher::mul(const Tensor& a, const Tensor& b) {
        const Shape out_shape = compute_broadcast_shape(a.shape(), b.shape());
        Tensor lhs = ensure_contiguous(a.broadcast_to(out_shape));
        Tensor rhs = ensure_contiguous(b.broadcast_to(out_shape));
        Tensor out(out_shape, a.dtype(), a.device());
        if (a.device().is_cpu())
            CPUBackend::mul(lhs.data_ptr(), rhs.data_ptr(), out.data_ptr(), out.numel());
        else
            throw std::runtime_error("Unsupported device");
        if (g_graph_builder) {
            g_graph_builder->build(
                OperationContext{.category = OpCategory::Binary, .type = OpType::Mul, .out = out, .inputs = {a, b}}
            );
        }
        return out;
    }

    Tensor Dispatcher::div(const Tensor& a, const Tensor& b) {
        const Shape out_shape = compute_broadcast_shape(a.shape(), b.shape());
        Tensor lhs = ensure_contiguous(a.broadcast_to(out_shape));
        Tensor rhs = ensure_contiguous(b.broadcast_to(out_shape));
        Tensor out(out_shape, a.dtype(), a.device());
        if (a.device().is_cpu())
            CPUBackend::div(lhs.data_ptr(), rhs.data_ptr(), out.data_ptr(), out.numel());
        else
            throw std::runtime_error("Unsupported device");
        if (g_graph_builder) {
            g_graph_builder->build(
                OperationContext{.category = OpCategory::Binary, .type = OpType::Div, .out = out, .inputs = {a, b}}
            );
        }
        return out;
    }

    Tensor Dispatcher::add_scalar(const Tensor& a, const float scalar) {
        Tensor lhs = ensure_contiguous(a);
        Tensor out(a.shape(), a.dtype(), a.device());
        if (a.device().is_cpu())
            CPUBackend::add_scalar(lhs.data_ptr(), scalar, out.data_ptr(), out.numel());
        else
            throw std::runtime_error("Unsupported device");
        // We omit graph builder for scalar ops for now, as they are primarily used in backward passes
        // where inputs don't require gradients.
        return out;
    }

    Tensor Dispatcher::sub_scalar(const Tensor& a, const float scalar) {
        Tensor lhs = ensure_contiguous(a);
        Tensor out(a.shape(), a.dtype(), a.device());
        if (a.device().is_cpu())
            CPUBackend::sub_scalar(lhs.data_ptr(), scalar, out.data_ptr(), out.numel());
        else
            throw std::runtime_error("Unsupported device");
        return out;
    }

    Tensor Dispatcher::mul_scalar(const Tensor& a, const float scalar) {
        Tensor lhs = ensure_contiguous(a);
        Tensor out(a.shape(), a.dtype(), a.device());
        if (a.device().is_cpu())
            CPUBackend::mul_scalar(lhs.data_ptr(), scalar, out.data_ptr(), out.numel());
        else
            throw std::runtime_error("Unsupported device");
        return out;
    }

    Tensor Dispatcher::div_scalar(const Tensor& a, const float scalar) {
        Tensor lhs = ensure_contiguous(a);
        Tensor out(a.shape(), a.dtype(), a.device());
        if (a.device().is_cpu())
            CPUBackend::div_scalar(lhs.data_ptr(), scalar, out.data_ptr(), out.numel());
        else
            throw std::runtime_error("Unsupported device");
        return out;
    }

    Tensor Dispatcher::neg(const Tensor& a) {
        Tensor lhs = ensure_contiguous(a);
        Tensor out(a.shape(), a.dtype(), a.device());
        if (a.device().is_cpu())
            CPUBackend::neg(lhs.data_ptr(), out.data_ptr(), out.numel());
        else
            throw std::runtime_error("Unsupported device");
        if (g_graph_builder) {
            g_graph_builder->build(
                OperationContext{.category = OpCategory::Unary, .type = OpType::Neg, .out = out, .inputs = {a}}
            );
        }
        return out;
    }

    Tensor Dispatcher::exp(const Tensor& a) {
        Tensor lhs = ensure_contiguous(a);
        Tensor out(a.shape(), a.dtype(), a.device());
        if (a.device().is_cpu())
            CPUBackend::exp(lhs.data_ptr(), out.data_ptr(), out.numel());
        else
            throw std::runtime_error("Unsupported device");
        if (g_graph_builder) {
            g_graph_builder->build(
                OperationContext{.category = OpCategory::Unary, .type = OpType::Exp, .out = out, .inputs = {a}}
            );
        }
        return out;
    }

    Tensor Dispatcher::tanh(const Tensor& a) {
        Tensor lhs = ensure_contiguous(a);
        Tensor out(a.shape(), a.dtype(), a.device());
        if (a.device().is_cpu())
            CPUBackend::tanh(lhs.data_ptr(), out.data_ptr(), out.numel());
        else
            throw std::runtime_error("Unsupported device");
        if (g_graph_builder) {
            g_graph_builder->build(
                OperationContext{.category = OpCategory::Unary, .type = OpType::Tanh, .out = out, .inputs = {a}}
            );
        }
        return out;
    }

    Tensor Dispatcher::log(const Tensor& a) {
        Tensor lhs = ensure_contiguous(a);
        Tensor out(a.shape(), a.dtype(), a.device());
        if (a.device().is_cpu())
            CPUBackend::log(lhs.data_ptr(), out.data_ptr(), out.numel());
        else
            throw std::runtime_error("Unsupported device");
        if (g_graph_builder) {
            g_graph_builder->build(
                OperationContext{.category = OpCategory::Unary, .type = OpType::Log, .out = out, .inputs = {a}}
            );
        }
        return out;
    }

    Tensor Dispatcher::sqrt(const Tensor& a) {
        Tensor lhs = ensure_contiguous(a);
        Tensor out(a.shape(), a.dtype(), a.device());
        if (a.device().is_cpu())
            CPUBackend::sqrt(lhs.data_ptr(), out.data_ptr(), out.numel());
        else
            throw std::runtime_error("Unsupported device");
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
        if (a.device().is_cpu())
            CPUBackend::relu(lhs.data_ptr(), out.data_ptr(), out.numel());
        else
            throw std::runtime_error("Unsupported device");
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
        if (grad_out.device().is_cpu())
            CPUBackend::relu_backward(lhs.data_ptr(), rhs.data_ptr(), out.data_ptr(), out.numel());
        else
            throw std::runtime_error("Unsupported device");
        return out;
    }

    Tensor Dispatcher::pow(const Tensor& a, float exponent) {
        Tensor lhs = ensure_contiguous(a);
        Tensor out(a.shape(), a.dtype(), a.device());
        if (a.device().is_cpu())
            CPUBackend::pow(lhs.data_ptr(), exponent, out.data_ptr(), out.numel());
        else
            throw std::runtime_error("Unsupported device");
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

        Tensor lhs = ensure_contiguous(a);
        const Tensor rhs = ensure_contiguous(b);

        const size_t M = a.shape()[0];
        const size_t K = a.shape()[1];
        const size_t N = b.shape()[1];

        Tensor out(Shape{M, N}, a.dtype(), a.device());

        if (a.device().is_cpu()) {
            CPUBackend::matmul(lhs.data_ptr(), rhs.data_ptr(), out.data_ptr(), M, K, N);
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
                CPUBackend::sum(lhs.data_ptr(), out.data_ptr(), 1, a.numel(), 1);
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
            CPUBackend::sum(lhs.data_ptr(), out.data_ptr(), outer_size, dim_size, inner_size);
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
        Tensor lhs = ensure_contiguous(a);

        if (!axis.has_value()) {
            const Shape out_shape = keepdim ? Shape(std::vector<size_t>(a.rank(), 1)) : Shape();
            Tensor out(out_shape, a.dtype(), a.device());
            if (a.device().is_cpu()) {
                CPUBackend::mean(lhs.data_ptr(), out.data_ptr(), 1, a.numel(), 1);
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
        Tensor out(out_shape, a.dtype(), a.device());

        size_t outer_size = 1;
        for (size_t i = 0; i < dim; ++i) outer_size *= a.shape()[i];
        const size_t dim_size = a.shape()[dim];
        size_t inner_size = 1;
        for (size_t i = dim + 1; i < a.rank(); ++i) inner_size *= a.shape()[i];

        if (a.device().is_cpu()) {
            CPUBackend::mean(lhs.data_ptr(), out.data_ptr(), outer_size, dim_size, inner_size);
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

        Tensor p_contig = ensure_contiguous(pred);
        Tensor t_contig = ensure_contiguous(target);

        const size_t N = pred.shape()[0];
        const size_t C = pred.shape()[1];

        Tensor out(Shape{}, pred.dtype(), pred.device());                   // scalar loss
        Tensor log_softmax_out(pred.shape(), pred.dtype(), pred.device());  // [N, C]

        if (pred.device().is_cpu()) {
            CPUBackend::cross_entropy(
                p_contig.data_ptr(), t_contig.data_ptr(), out.data_ptr(), log_softmax_out.data_ptr(), N, C
            );
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

        if (param.has_internal_overlap()) {
            throw std::runtime_error("sgd: in-place operation on a tensor with overlapping memory is not supported.");
        }

        const bool is_aliased =
            (param.impl()->storage() == grad.impl()->storage()) &&
            (param.data_ptr() != grad.data_ptr() || param.stride() != grad.stride() || param.shape() != grad.shape());
        Tensor safe_grad = is_aliased ? grad.clone() : grad;

        if (param.device().is_cpu()) {
            if (param.is_contiguous() && safe_grad.is_contiguous()) {
                CPUBackend::sgd(param.data_ptr(), safe_grad.data_ptr(), lr, param.numel());
            } else if (param.rank() == 2) {
                const size_t rows = param.shape()[0];
                const size_t cols = param.shape()[1];
                const size_t p_stride0 = param.stride()[0];
                const size_t p_stride1 = param.stride()[1];
                const size_t g_stride0 = safe_grad.stride()[0];
                const size_t g_stride1 = safe_grad.stride()[1];
                float* p_data = param.data_ptr();
                const float* g_data = safe_grad.data_ptr();

#pragma omp parallel for
                for (ptrdiff_t r = 0; r < static_cast<ptrdiff_t>(rows); ++r) {
#pragma omp simd
                    for (size_t c = 0; c < cols; ++c) {
                        p_data[r * p_stride0 + c * p_stride1] -= lr * g_data[r * g_stride0 + c * g_stride1];
                    }
                }
            } else {
                float* p_data = param.data_ptr();
                const float* g_data = safe_grad.data_ptr();
                const size_t total_elements = param.numel();

#pragma omp parallel
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
                        size_t offset_p = it.compute_offset(param.stride());
                        size_t offset_g = it.compute_offset(safe_grad.stride());

                        for (size_t i = start; i < end; ++i) {
                            p_data[offset_p] -= lr * g_data[offset_g];
                            it.advance(offset_p, param.stride(), offset_g, safe_grad.stride());
                        }
                    }
                }
            }
        } else {
            throw std::runtime_error("Unsupported device");
        }

        param.increment_version();
    }

}  // namespace helix
