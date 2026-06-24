#pragma once

#include <optional>

#include "core/tensor.hpp"

namespace helix {

    class GraphBuilderInterface;

    class Dispatcher {
    public:
        static void register_graph_builder(GraphBuilderInterface* builder);
        static GraphBuilderInterface* get_graph_builder();

        // Ensure the tensor is contiguous. If not, returns a contiguous clone.
        static Tensor ensure_contiguous(const Tensor& t);

        // Mathematical Operations
        static Tensor add(const Tensor& a, const Tensor& b);
        static Tensor sub(const Tensor& a, const Tensor& b);
        static Tensor mul(const Tensor& a, const Tensor& b);
        static Tensor div(const Tensor& a, const Tensor& b);

        static Tensor add_scalar(const Tensor& a, float scalar);
        static Tensor sub_scalar(const Tensor& a, float scalar);
        static Tensor mul_scalar(const Tensor& a, float scalar);
        static Tensor div_scalar(const Tensor& a, float scalar);
        static Tensor matmul(const Tensor& a, const Tensor& b);

        // Unary Operations
        static Tensor neg(const Tensor& a);
        static Tensor exp(const Tensor& a);
        static Tensor log(const Tensor& a);
        static Tensor sqrt(const Tensor& a);
        static Tensor pow(const Tensor& a, float exponent);

        static Tensor sum(const Tensor& a, std::optional<size_t> axis = std::nullopt, bool keepdim = false);
        static Tensor mean(const Tensor& a, std::optional<size_t> axis = std::nullopt, bool keepdim = false);
    };

}  // namespace helix
