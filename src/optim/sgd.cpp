#include "optim/sgd.hpp"

#include <stdexcept>

#include "core/dispatcher.hpp"

namespace helix {

    SGD::SGD(std::vector<Tensor> params, const float lr) : Optimizer(std::move(params)), learning_rate_(lr) {}

    void SGD::step() {
        for (auto& param : params_) {
            // 1. requires_grad
            if (!param.requires_grad()) {
                continue;
            }

            // 2. grad exists
            if (!param.has_grad()) {
                throw std::runtime_error("Gradient does not exist for parameter");
            }

            const auto& grad = param.grad();

            // 3. shape identical
            if (param.shape() != grad.shape()) {
                throw std::invalid_argument("Parameter and gradient shapes do not match");
            }

            // 4. dtype identical
            if (param.dtype() != grad.dtype()) {
                throw std::invalid_argument("Parameter and gradient dtypes do not match");
            }

            // 5. device identical
            if (param.device() != grad.device()) {
                throw std::invalid_argument("Parameter and gradient devices do not match");
            }

            // Route execution to Dispatcher
            Dispatcher::sgd(param, grad, learning_rate_);
        }
    }

}  // namespace helix
