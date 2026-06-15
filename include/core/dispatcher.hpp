#pragma once

#include <optional>

#include "core/tensor.hpp"

namespace helix {

    class Dispatcher {
    public:
        // Ensure the tensor is contiguous. If not, returns a contiguous clone.
        static Tensor ensure_contiguous(const Tensor& t);

        // Mathematical Operations
        static Tensor add(const Tensor& a, const Tensor& b);
        static Tensor sub(const Tensor& a, const Tensor& b);
        static Tensor mul(const Tensor& a, const Tensor& b);
        static Tensor div(const Tensor& a, const Tensor& b);
        static Tensor matmul(const Tensor& a, const Tensor& b);

        // Unary Operations
        static Tensor neg(const Tensor& a);

        static Tensor sum(const Tensor& a, std::optional<size_t> axis = std::nullopt, bool keepdim = false);
        static Tensor mean(const Tensor& a, std::optional<size_t> axis = std::nullopt, bool keepdim = false);
    };

}  // namespace helix
