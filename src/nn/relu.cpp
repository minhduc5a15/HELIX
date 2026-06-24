#include "nn/relu.hpp"

namespace helix {

    Tensor relu(const Tensor& x) { return x.relu(); }

    Tensor ReLU::forward(const Tensor& input) { return relu(input); }

}  // namespace helix
