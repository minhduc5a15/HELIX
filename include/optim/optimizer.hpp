#pragma once

#include <vector>

#include "core/tensor.hpp"

namespace helix {

    class Optimizer {
    protected:
        std::vector<Tensor> params_;

    public:
        explicit Optimizer(std::vector<Tensor> params) : params_(std::move(params)) {}
        virtual ~Optimizer() = default;

        virtual void step() = 0;

        void zero_grad() {
            for (auto& param : params_) {
                if (param.requires_grad() && param.has_grad()) {
                    param.grad().zero_();
                }
            }
        }

        const std::vector<Tensor>& parameters() const { return params_; }
    };

}  // namespace helix
