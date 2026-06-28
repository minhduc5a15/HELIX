#include "autograd/function.hpp"

#include "autograd/function_utils.hpp"
#include "core/dispatcher.hpp"

namespace helix {

    Tensor sum_to_shape(const Tensor& grad, const Shape& target_shape) {
        if (grad.shape() == target_shape) {
            return grad;
        }

        Tensor result = grad.detach();

        // 1. Reduce leading broadcasted dimensions (if grad has higher rank)
        while (result.rank() > target_shape.rank()) {
            result = result.sum(0, false);  // sum along axis 0, not keeping dim
        }

        // 2. Reduce broadcasted dimensions where target shape is 1 but grad is > 1
        for (size_t i = 0; i < target_shape.rank(); ++i) {
            if (target_shape[i] == 1 && result.shape()[i] > 1) {
                result = result.sum(i, true);  // sum along axis i, keep dim
            }
        }

        return result;
    }

    std::vector<Tensor> AddBackward::backward(const std::vector<Tensor>& grad_outputs) {
        // d(A+B)/dA = 1, d(A+B)/dB = 1.
        Tensor grad_a = sum_to_shape(grad_outputs[0], shape_a_);
        Tensor grad_b = sum_to_shape(grad_outputs[0], shape_b_);
        return {grad_a, grad_b};
    }

    std::vector<Tensor> SubBackward::backward(const std::vector<Tensor>& grad_outputs) {
        // d(A-B)/dA = 1, d(A-B)/dB = -1
        Tensor grad_a = sum_to_shape(grad_outputs[0], shape_a_);
        Tensor grad_b = sum_to_shape(-grad_outputs[0], shape_b_);
        return {grad_a, grad_b};
    }

    std::vector<Tensor> MulBackward::backward(const std::vector<Tensor>& grad_outputs) {
        // d(A*B)/dA = B, d(A*B)/dB = A
        const Tensor& a = saved_a_.unpack();
        const Tensor& b = saved_b_.unpack();
        Tensor grad_a = sum_to_shape(grad_outputs[0] * b, a.shape());
        Tensor grad_b = sum_to_shape(grad_outputs[0] * a, b.shape());
        return {grad_a, grad_b};
    }

    std::vector<Tensor> DivBackward::backward(const std::vector<Tensor>& grad_outputs) {
        // d(A/B)/dA = 1/B, d(A/B)/dB = -A / B^2
        const Tensor& a = saved_a_.unpack();
        const Tensor& b = saved_b_.unpack();
        Tensor grad_a = sum_to_shape(grad_outputs[0] / b, a.shape());
        Tensor grad_b = sum_to_shape(-grad_outputs[0] * a / (b * b), b.shape());
        return {grad_a, grad_b};
    }

    std::vector<Tensor> MatMulBackward::backward(const std::vector<Tensor>& grad_outputs) {
        // d(A@B)/dA = grad_output @ B^T
        // d(A@B)/dB = A^T @ grad_output
        const Tensor& a = saved_a_.unpack();
        const Tensor& b = saved_b_.unpack();

        const size_t rank_a = a.rank();
        const size_t rank_b = b.rank();

        const Tensor b_t = b.transpose(rank_b - 2, rank_b - 1);
        const Tensor a_t = a.transpose(rank_a - 2, rank_a - 1);

        return {grad_outputs[0].matmul(b_t), a_t.matmul(grad_outputs[0])};
    }

    std::vector<Tensor> NegBackward::backward(const std::vector<Tensor>& grad_outputs) { return {-grad_outputs[0]}; }

    std::vector<Tensor> ExpBackward::backward(const std::vector<Tensor>& grad_outputs) {
        const Tensor& out = saved_out_.unpack();
        return {grad_outputs[0] * out};
    }

    std::vector<Tensor> LogBackward::backward(const std::vector<Tensor>& grad_outputs) {
        const Tensor& a = saved_a_.unpack();
        return {grad_outputs[0] / a};
    }

    std::vector<Tensor> SqrtBackward::backward(const std::vector<Tensor>& grad_outputs) {
        const Tensor& out = saved_out_.unpack();
        // d(sqrt(x))/dx = 1 / (2 * sqrt(x))
        return {grad_outputs[0] / (out * 2.0f)};
    }

    PowBackward::PowBackward(const Tensor& a, const float exponent) : saved_a_(a), exponent_(exponent) {}

    std::vector<Tensor> PowBackward::backward(const std::vector<Tensor>& grad_outputs) {
        const Tensor& a = saved_a_.unpack();
        // n * x^(n-1)
        const Tensor a_pow = a.pow(exponent_ - 1.0f);
        return {grad_outputs[0] * exponent_ * a_pow};
    }

    ReLUBackward::ReLUBackward(const Tensor& a) : saved_a_(a) {}

    std::vector<Tensor> ReLUBackward::backward(const std::vector<Tensor>& grad_outputs) {
        const Tensor& a = saved_a_.unpack();
        return {Dispatcher::relu_backward(grad_outputs[0], a)};
    }

    std::vector<Tensor> SumBackward::backward(const std::vector<Tensor>& grad_outputs) {
        Tensor grad = grad_outputs[0];

        // If reduced along an axis and keepdim is false, we must unsqueeze it back
        // to its broadcastable shape before calling broadcast_to
        if (axis_.has_value() && !keepdim_) {
            const size_t dim = axis_.value();
            std::vector<size_t> unsqueezed_dims = grad.shape().vec();
            unsqueezed_dims.insert(unsqueezed_dims.begin() + dim, 1);
            grad = grad.reshape(Shape(unsqueezed_dims));
        }

        return {grad.broadcast_to(input_shape_)};
    }

    std::vector<Tensor> MeanBackward::backward(const std::vector<Tensor>& grad_outputs) {
        Tensor grad = grad_outputs[0];

        if (axis_.has_value() && !keepdim_) {
            const size_t dim = axis_.value();
            std::vector<size_t> unsqueezed_dims = grad.shape().vec();
            unsqueezed_dims.insert(unsqueezed_dims.begin() + dim, 1);
            grad = grad.reshape(Shape(unsqueezed_dims));
        }

        grad = grad.broadcast_to(input_shape_);

        // Calculate reduction size
        const float reduction_size =
            static_cast<float>(input_shape_.numel()) / static_cast<float>(grad_outputs[0].numel());

        // Divide straight by N using the new scalar operator
        return {grad / reduction_size};
    }

}  // namespace helix
