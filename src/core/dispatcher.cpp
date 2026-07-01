#include "core/dispatcher.hpp"

#include <stdexcept>

#include "backend/cpu_backend.hpp"
#include "core/broadcast.hpp"
#include "core/graph_builder.hpp"
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
        Tensor lhs = ensure_contiguous(a);
        Tensor rhs = ensure_contiguous(b);
        if (a.device().is_cpu())
            CPUBackend::add(lhs.data_ptr(), rhs.data_ptr(), lhs.data_ptr(), lhs.numel());
        else
            throw std::runtime_error("Unsupported device");

        if (!a.is_contiguous()) {
            a.copy_(lhs);
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
            g_graph_builder->build(OperationContext{OpCategory::Binary, OpType::Sub, out, {a, b}});
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
            g_graph_builder->build(OperationContext{OpCategory::Binary, OpType::Mul, out, {a, b}});
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
            g_graph_builder->build(OperationContext{OpCategory::Binary, OpType::Div, out, {a, b}});
        }
        return out;
    }

    Tensor Dispatcher::add_scalar(const Tensor& a, float scalar) {
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

    Tensor Dispatcher::sub_scalar(const Tensor& a, float scalar) {
        Tensor lhs = ensure_contiguous(a);
        Tensor out(a.shape(), a.dtype(), a.device());
        if (a.device().is_cpu())
            CPUBackend::sub_scalar(lhs.data_ptr(), scalar, out.data_ptr(), out.numel());
        else
            throw std::runtime_error("Unsupported device");
        return out;
    }

    Tensor Dispatcher::mul_scalar(const Tensor& a, float scalar) {
        Tensor lhs = ensure_contiguous(a);
        Tensor out(a.shape(), a.dtype(), a.device());
        if (a.device().is_cpu())
            CPUBackend::mul_scalar(lhs.data_ptr(), scalar, out.data_ptr(), out.numel());
        else
            throw std::runtime_error("Unsupported device");
        return out;
    }

    Tensor Dispatcher::div_scalar(const Tensor& a, float scalar) {
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
            g_graph_builder->build(OperationContext{OpCategory::Unary, OpType::Neg, out, {a}});
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
            g_graph_builder->build(OperationContext{OpCategory::Unary, OpType::Exp, out, {a}});
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
            g_graph_builder->build(OperationContext{OpCategory::Unary, OpType::Log, out, {a}});
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
            g_graph_builder->build(OperationContext{OpCategory::Unary, OpType::Sqrt, out, {a}});
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
            g_graph_builder->build(OperationContext{OpCategory::Unary, OpType::ReLU, out, {a}});
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
            OperationContext ctx{OpCategory::Unary, OpType::Pow, out, {a}};
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

        // Transpose B and ensure it's contiguous to improve cache locality during the dot product loop
        Tensor rhs_t = ensure_contiguous(rhs.transpose(0, 1));

        const size_t M = a.shape()[0];
        const size_t K = a.shape()[1];
        const size_t N = b.shape()[1];

        Tensor out(Shape{M, N}, a.dtype(), a.device());

        if (a.device().is_cpu()) {
            CPUBackend::matmul(lhs.data_ptr(), rhs_t.data_ptr(), out.data_ptr(), M, K, N);
        } else {
            throw std::runtime_error("Unsupported device");
        }
        if (g_graph_builder) {
            g_graph_builder->build(OperationContext{OpCategory::Matrix, OpType::MatMul, out, {a, b}});
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
                OperationContext ctx{OpCategory::Reduce, OpType::Sum, out, {a}};
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
            OperationContext ctx{OpCategory::Reduce, OpType::Sum, out, {a}};
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
                OperationContext ctx{OpCategory::Reduce, OpType::Mean, out, {a}};
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
            OperationContext ctx{OpCategory::Reduce, OpType::Mean, out, {a}};
            ctx.attributes["axis"] = axis;
            ctx.attributes["keepdim"] = keepdim;
            g_graph_builder->build(ctx);
        }
        return out;
    }

    void Dispatcher::sgd(Tensor& param, const Tensor& grad, float lr) {
        Tensor lhs = ensure_contiguous(param);
        Tensor rhs = ensure_contiguous(grad);
        if (param.device().is_cpu()) {
            CPUBackend::sgd(lhs.data_ptr(), rhs.data_ptr(), lr, lhs.numel());
            if (!param.is_contiguous()) {
                param.copy_(lhs);
            }
        } else {
            throw std::runtime_error("Unsupported device");
        }
    }

}  // namespace helix
