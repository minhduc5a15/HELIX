#include "core/dispatcher.hpp"

#include <stdexcept>

#include "backend/cpu_backend.hpp"
#include "core/broadcast.hpp"

namespace helix {

    Tensor Dispatcher::ensure_contiguous(const Tensor& t) { return t.contiguous(); }

    // TODO:
    // Current CPU backend only supports contiguous tensors.
    // Once TensorIterator is implemented,
    // remove these contiguous() calls.
    Tensor Dispatcher::add(const Tensor& a, const Tensor& b) {
        Shape out_shape = compute_broadcast_shape(a.shape(), b.shape());
        Tensor lhs = ensure_contiguous(a.broadcast_to(out_shape));
        Tensor rhs = ensure_contiguous(b.broadcast_to(out_shape));
        Tensor out(out_shape, a.dtype(), a.device());
        if (a.device().is_cpu())
            CPUBackend::add(lhs.data_ptr(), rhs.data_ptr(), out.data_ptr(), out.numel());
        else
            throw std::runtime_error("Unsupported device");
        return out;
    }

    Tensor Dispatcher::sub(const Tensor& a, const Tensor& b) {
        Shape out_shape = compute_broadcast_shape(a.shape(), b.shape());
        Tensor lhs = ensure_contiguous(a.broadcast_to(out_shape));
        Tensor rhs = ensure_contiguous(b.broadcast_to(out_shape));
        Tensor out(out_shape, a.dtype(), a.device());
        if (a.device().is_cpu())
            CPUBackend::sub(lhs.data_ptr(), rhs.data_ptr(), out.data_ptr(), out.numel());
        else
            throw std::runtime_error("Unsupported device");
        return out;
    }

    Tensor Dispatcher::mul(const Tensor& a, const Tensor& b) {
        Shape out_shape = compute_broadcast_shape(a.shape(), b.shape());
        Tensor lhs = ensure_contiguous(a.broadcast_to(out_shape));
        Tensor rhs = ensure_contiguous(b.broadcast_to(out_shape));
        Tensor out(out_shape, a.dtype(), a.device());
        if (a.device().is_cpu())
            CPUBackend::mul(lhs.data_ptr(), rhs.data_ptr(), out.data_ptr(), out.numel());
        else
            throw std::runtime_error("Unsupported device");
        return out;
    }

    Tensor Dispatcher::div(const Tensor& a, const Tensor& b) {
        Shape out_shape = compute_broadcast_shape(a.shape(), b.shape());
        Tensor lhs = ensure_contiguous(a.broadcast_to(out_shape));
        Tensor rhs = ensure_contiguous(b.broadcast_to(out_shape));
        Tensor out(out_shape, a.dtype(), a.device());
        if (a.device().is_cpu())
            CPUBackend::div(lhs.data_ptr(), rhs.data_ptr(), out.data_ptr(), out.numel());
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
        Tensor rhs = ensure_contiguous(b);

        // Transpose B and ensure it's contiguous to improve cache locality during the dot product loop
        Tensor rhs_t = ensure_contiguous(rhs.transpose(0, 1));

        size_t M = a.shape()[0];
        size_t K = a.shape()[1];
        size_t N = b.shape()[1];

        Tensor out(Shape{M, N}, a.dtype(), a.device());

        if (a.device().is_cpu()) {
            CPUBackend::matmul(lhs.data_ptr(), rhs_t.data_ptr(), out.data_ptr(), M, K, N);
        } else {
            throw std::runtime_error("Unsupported device");
        }
        return out;
    }

    Tensor Dispatcher::sum(const Tensor& a, std::optional<size_t> axis, bool keepdim) {
        Tensor lhs = ensure_contiguous(a);

        if (!axis.has_value()) {
            Shape out_shape = keepdim ? Shape(std::vector<size_t>(a.rank(), 1)) : Shape();
            Tensor out(out_shape, a.dtype(), a.device());
            if (a.device().is_cpu()) {
                CPUBackend::sum(lhs.data_ptr(), out.data_ptr(), 1, a.numel(), 1);
            } else {
                throw std::runtime_error("Unsupported device");
            }
            return out;
        }

        size_t dim = axis.value();
        if (dim >= a.rank()) throw std::out_of_range("axis out of bounds");

        std::vector<size_t> out_dims;
        for (size_t i = 0; i < a.rank(); ++i) {
            if (i == dim) {
                if (keepdim) out_dims.push_back(1);
            } else {
                out_dims.push_back(a.shape()[i]);
            }
        }
        Shape out_shape(out_dims);
        Tensor out(out_shape, a.dtype(), a.device());

        size_t outer_size = 1;
        for (size_t i = 0; i < dim; ++i) outer_size *= a.shape()[i];
        size_t dim_size = a.shape()[dim];
        size_t inner_size = 1;
        for (size_t i = dim + 1; i < a.rank(); ++i) inner_size *= a.shape()[i];

        if (a.device().is_cpu()) {
            CPUBackend::sum(lhs.data_ptr(), out.data_ptr(), outer_size, dim_size, inner_size);
        } else {
            throw std::runtime_error("Unsupported device");
        }
        return out;
    }

    Tensor Dispatcher::mean(const Tensor& a, std::optional<size_t> axis, bool keepdim) {
        Tensor lhs = ensure_contiguous(a);

        if (!axis.has_value()) {
            Shape out_shape = keepdim ? Shape(std::vector<size_t>(a.rank(), 1)) : Shape();
            Tensor out(out_shape, a.dtype(), a.device());
            if (a.device().is_cpu()) {
                CPUBackend::mean(lhs.data_ptr(), out.data_ptr(), 1, a.numel(), 1);
            } else {
                throw std::runtime_error("Unsupported device");
            }
            return out;
        }

        size_t dim = axis.value();
        if (dim >= a.rank()) throw std::out_of_range("axis out of bounds");

        std::vector<size_t> out_dims;
        for (size_t i = 0; i < a.rank(); ++i) {
            if (i == dim) {
                if (keepdim) out_dims.push_back(1);
            } else {
                out_dims.push_back(a.shape()[i]);
            }
        }
        Shape out_shape(out_dims);
        Tensor out(out_shape, a.dtype(), a.device());

        size_t outer_size = 1;
        for (size_t i = 0; i < dim; ++i) outer_size *= a.shape()[i];
        size_t dim_size = a.shape()[dim];
        size_t inner_size = 1;
        for (size_t i = dim + 1; i < a.rank(); ++i) inner_size *= a.shape()[i];

        if (a.device().is_cpu()) {
            CPUBackend::mean(lhs.data_ptr(), out.data_ptr(), outer_size, dim_size, inner_size);
        } else {
            throw std::runtime_error("Unsupported device");
        }
        return out;
    }

}  // namespace helix
