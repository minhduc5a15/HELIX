#include "nn/loss.hpp"

#include "core/dispatcher.hpp"

namespace helix {

    Tensor mse_loss(const Tensor& pred, const Tensor& target) {
        const Tensor diff = pred - target;
        return (diff * diff).mean();
    }

    Tensor cross_entropy_loss(const Tensor& pred, const Tensor& target) {
        return Dispatcher::cross_entropy(pred, target);
    }

}  // namespace helix
