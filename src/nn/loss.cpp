#include "nn/loss.hpp"

namespace helix {

    Tensor mse_loss(const Tensor& pred, const Tensor& target) {
        Tensor diff = pred - target;
        return (diff * diff).mean();
    }

}  // namespace helix
