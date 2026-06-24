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
        AddBackward(Shape shape_a, Shape shape_b) : shape_a_(std::move(shape_a)), shape_b_(std::move(shape_b)) {}
        std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

    private:
        Shape shape_a_;
        Shape shape_b_;
    };

    class SubBackward : public Node {
    public:
        SubBackward(Shape shape_a, Shape shape_b) : shape_a_(std::move(shape_a)), shape_b_(std::move(shape_b)) {}
        std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

    private:
        Shape shape_a_;
        Shape shape_b_;
    };

    class MulBackward : public Node {
    public:
        MulBackward(const Tensor& a, const Tensor& b) : saved_a_(a), saved_b_(b) {}
        std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

    private:
        SavedTensor saved_a_;
        SavedTensor saved_b_;
    };

    class DivBackward : public Node {
    public:
        DivBackward(const Tensor& a, const Tensor& b) : saved_a_(a), saved_b_(b) {}
        std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

    private:
        SavedTensor saved_a_;
        SavedTensor saved_b_;
    };

    class MatMulBackward : public Node {
    public:
        MatMulBackward(const Tensor& a, const Tensor& b) : saved_a_(a), saved_b_(b) {}
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

    class ExpBackward : public Node {
    public:
        ExpBackward(const Tensor& out) : saved_out_(out) {}
        std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

    private:
        SavedTensor saved_out_;
    };

    class LogBackward : public Node {
    public:
        LogBackward(const Tensor& a) : saved_a_(a) {}
        std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

    private:
        SavedTensor saved_a_;
    };

    class SqrtBackward : public Node {
    public:
        SqrtBackward(const Tensor& out) : saved_out_(out) {}
        std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

    private:
        SavedTensor saved_out_;
    };

    class PowBackward : public Node {
    public:
        PowBackward(const Tensor& a, float exponent);
        std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

    private:
        SavedTensor saved_a_;
        float exponent_;
    };

    class ReLUBackward : public Node {
    public:
        ReLUBackward(const Tensor& a);
        std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

    private:
        SavedTensor saved_a_;
    };

    class SumBackward : public Node {
    public:
        SumBackward(Shape input_shape, std::optional<size_t> axis, bool keepdim)
            : input_shape_(std::move(input_shape)), axis_(axis), keepdim_(keepdim) {}
        std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

    private:
        Shape input_shape_;
        std::optional<size_t> axis_;
        bool keepdim_;
    };

    class MeanBackward : public Node {
    public:
        MeanBackward(Shape input_shape, std::optional<size_t> axis, bool keepdim)
            : input_shape_(std::move(input_shape)), axis_(axis), keepdim_(keepdim) {}
        std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override;

    private:
        Shape input_shape_;
        std::optional<size_t> axis_;
        bool keepdim_;
    };

}  // namespace helix
