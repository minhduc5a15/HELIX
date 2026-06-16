#pragma once

#include <vector>
#include "autograd/node.hpp"
#include "core/tensor.hpp"

namespace helix {

    // Helper to store tensors safely (avoiding reference cycles).
    // By calling detach(), we create a new TensorImpl sharing the storage
    // but without the autograd_meta_ history, thus breaking the cycle:
    // Tensor -> AutogradMeta -> Node -> SavedTensor -> Tensor.
    class SavedTensor {
    public:
        SavedTensor(const Tensor& t) : tensor_(t.detach()) {}
        const Tensor& unpack() const { return tensor_; }
    private:
        Tensor tensor_;
    };

    class AddBackward : public Node {
    public:
        AddBackward() = default;
        std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    };

    class SubBackward : public Node {
    public:
        SubBackward() = default;
        std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    };

    class MulBackward : public Node {
    public:
        MulBackward(const Tensor& a, const Tensor& b) 
            : saved_a_(a), saved_b_(b) {}
        std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    private:
        SavedTensor saved_a_;
        SavedTensor saved_b_;
    };

    class DivBackward : public Node {
    public:
        DivBackward(const Tensor& a, const Tensor& b) 
            : saved_a_(a), saved_b_(b) {}
        std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    private:
        SavedTensor saved_a_;
        SavedTensor saved_b_;
    };

    class MatMulBackward : public Node {
    public:
        MatMulBackward(const Tensor& a, const Tensor& b) 
            : saved_a_(a), saved_b_(b) {}
        std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    private:
        SavedTensor saved_a_;
        SavedTensor saved_b_;
    };

    class NegBackward : public Node {
    public:
        NegBackward() = default;
        std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    };

    class SumBackward : public Node {
    public:
        SumBackward(Shape input_shape) : input_shape_(std::move(input_shape)) {}
        std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    private:
        Shape input_shape_;
    };

    class MeanBackward : public Node {
    public:
        MeanBackward(Shape input_shape) : input_shape_(std::move(input_shape)) {}
        std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;
    private:
        Shape input_shape_;
    };

}  // namespace helix
