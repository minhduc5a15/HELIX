#pragma once

#include <functional>
#include <vector>

#include "core/tensor.hpp"

namespace helix {

    // gradient_check verifies if the autograd engine computes the gradient correctly
    // by comparing it with the numerical gradient (using finite differences).
    // func MUST return a scalar tensor.
    bool gradient_check(
        const std::function<Tensor(const std::vector<Tensor>&)>& func,
        const std::vector<Tensor>& inputs,
        float eps = 1e-3f,
        float tolerance = 1e-3f
    );

}  // namespace helix
