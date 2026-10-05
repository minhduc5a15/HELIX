#include "nn/flatten.hpp"

namespace helix {

    Flatten::Flatten(int64_t start_dim, int64_t end_dim) : start_dim_(start_dim), end_dim_(end_dim) {}

    Tensor Flatten::forward(const Tensor& input) { return input.flatten(start_dim_, end_dim_); }

}  // namespace helix
