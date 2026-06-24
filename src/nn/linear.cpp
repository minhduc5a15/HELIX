#include "nn/linear.hpp"

#include <cmath>

namespace helix {

    Linear::Linear(size_t in_features, size_t out_features) {
        // scaled random initialization for weight (LeCun Initialization)
        weight_ = Tensor::randn({in_features, out_features}) / std::sqrt(static_cast<float>(in_features));
        // initialize bias to zero
        bias_ = Tensor::zeros({out_features});

        weight_.set_requires_grad(true);
        bias_.set_requires_grad(true);
    }

    Tensor Linear::forward(const Tensor& input) {
        // y = xW + b
        return input.matmul(weight_) + bias_;
    }

    std::vector<std::pair<std::string, Tensor>> Linear::named_parameters() {
        return {{"weight", weight_}, {"bias", bias_}};
    }

}  // namespace helix
