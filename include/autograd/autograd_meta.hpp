#pragma once

#include <memory>
#include "core/tensor.hpp"
#include "autograd/node.hpp"

namespace helix {

    // Actual implementation of AutogradMeta.
    // Core holds an opaque pointer to this class.
    class AutogradMeta {
    public:
        AutogradMeta(bool requires_grad = false) 
            : requires_grad_(requires_grad) {}

        bool requires_grad() const { return requires_grad_; }
        void set_requires_grad(bool req) { requires_grad_ = req; }

        Tensor& grad() { return grad_; }
        const Tensor& grad() const { return grad_; }
        void set_grad(const Tensor& g) { grad_ = g; }

        std::shared_ptr<Node> grad_fn() const { return grad_fn_; }
        void set_grad_fn(std::shared_ptr<Node> fn) { grad_fn_ = std::move(fn); }

        std::shared_ptr<Node> grad_accumulator() const { return grad_accumulator_; }
        void set_grad_accumulator(std::shared_ptr<Node> acc) { grad_accumulator_ = std::move(acc); }

    private:
        bool requires_grad_;
        Tensor grad_;
        std::shared_ptr<Node> grad_fn_;
        std::shared_ptr<Node> grad_accumulator_;
    };

}  // namespace helix
