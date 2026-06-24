#pragma once

#include "nn/module.hpp"

namespace helix {

    Tensor relu(const Tensor& x);

    class ReLU : public Module {
    public:
        ReLU() = default;
        Tensor forward(const Tensor& input) override;
    };

}  // namespace helix
