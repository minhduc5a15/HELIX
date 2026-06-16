#include "autograd/function.hpp"

namespace helix {

    // Helper function to reduce gradient if it was broadcasted
    // A proper implementation requires checking shapes and summing over broadcasted dims.
    static Tensor reduce_gradient(Tensor grad, const Shape& target_shape) {
        if (grad.shape() == target_shape) {
            return grad;
        }
        // TODO: Full implementation of broadcasting backward sum over axes.
        // For basic functionality in week 1 without a generic sum along multiple axes,
        // we return the gradient as is (will only work correctly if no broadcasting occurred).
        return grad;
    }

    std::vector<Tensor> AddBackward::backward(const std::vector<Tensor>& grad_outputs) {
        // d(A+B)/dA = 1, d(A+B)/dB = 1.
        return {grad_outputs[0], grad_outputs[0]};
    }

    std::vector<Tensor> SubBackward::backward(const std::vector<Tensor>& grad_outputs) {
        // d(A-B)/dA = 1, d(A-B)/dB = -1
        return {grad_outputs[0], -grad_outputs[0]};
    }

    std::vector<Tensor> MulBackward::backward(const std::vector<Tensor>& grad_outputs) {
        // d(A*B)/dA = B, d(A*B)/dB = A
        const Tensor& a = saved_a_.unpack();
        const Tensor& b = saved_b_.unpack();
        return {grad_outputs[0] * b, grad_outputs[0] * a};
    }

    std::vector<Tensor> DivBackward::backward(const std::vector<Tensor>& grad_outputs) {
        // d(A/B)/dA = 1/B, d(A/B)/dB = -A / B^2
        const Tensor& a = saved_a_.unpack();
        const Tensor& b = saved_b_.unpack();
        Tensor grad_a = grad_outputs[0] / b;
        Tensor grad_b = -grad_outputs[0] * a / (b * b);
        return {grad_a, grad_b};
    }

    std::vector<Tensor> MatMulBackward::backward(const std::vector<Tensor>& grad_outputs) {
        // d(A@B)/dA = grad_output @ B^T
        // d(A@B)/dB = A^T @ grad_output
        const Tensor& a = saved_a_.unpack();
        const Tensor& b = saved_b_.unpack();
        
        size_t rank_a = a.rank();
        size_t rank_b = b.rank();
        
        Tensor b_t = b.transpose(rank_b - 2, rank_b - 1);
        Tensor a_t = a.transpose(rank_a - 2, rank_a - 1);

        return {grad_outputs[0].matmul(b_t), a_t.matmul(grad_outputs[0])};
    }

    std::vector<Tensor> NegBackward::backward(const std::vector<Tensor>& grad_outputs) {
        return {-grad_outputs[0]};
    }

    std::vector<Tensor> SumBackward::backward(const std::vector<Tensor>& grad_outputs) {
        // Un-reduce gradient. grad_output broadcasted to input_shape.
        return {grad_outputs[0].broadcast_to(input_shape_)};
    }

    std::vector<Tensor> MeanBackward::backward(const std::vector<Tensor>& grad_outputs) {
        float n = static_cast<float>(input_shape_.numel() / grad_outputs[0].numel());
        Tensor grad = grad_outputs[0].broadcast_to(input_shape_);
        
        // Multiply by 1/n
        std::vector<float> data(grad.numel(), 1.0f / n);
        Tensor div_tensor(data, grad.shape());
        return {grad * div_tensor};
    }

}  // namespace helix
