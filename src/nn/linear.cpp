#include "nn/linear.hpp"

#include <cmath>

namespace helix {

    Linear::Linear(size_t in_features, size_t out_features) {
        if (in_features == 0 || out_features == 0) {
            throw std::invalid_argument("Linear in_features and out_features must be greater than 0");
        }
        weight_ = Tensor::randn({in_features, out_features}) / std::sqrt(static_cast<float>(in_features));
        bias_ = Tensor::zeros({out_features}, weight_.dtype(), weight_.device());
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
