#pragma once

#include <optional>

#include "core/tensor.hpp"

namespace helix {

    class GraphBuilderInterface;

    constexpr size_t OMP_THRESHOLD = 262144;
    constexpr size_t OMP_NON_CONTIGUOUS_THRESHOLD = 262144;

    class Dispatcher {
    public:
        static void register_graph_builder(GraphBuilderInterface* builder);
        static GraphBuilderInterface* get_graph_builder();

        // Ensure the tensor is contiguous. If not, returns a contiguous clone.
        static Tensor ensure_contiguous(const Tensor& t);

        // View Operations
        static Tensor clone(const Tensor& a);
        static Tensor view(const Tensor& a, Shape new_shape);
        static Tensor slice(const Tensor& a, size_t dim, size_t start, size_t end);
        static Tensor transpose(const Tensor& a, size_t dim0, size_t dim1);
        static Tensor broadcast_to(const Tensor& a, Shape new_shape);

        // Mathematical Operations
        static Tensor add(const Tensor& a, const Tensor& b);
        static void add_(Tensor& a, const Tensor& b);
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
        static Tensor tanh(const Tensor& a);
        static Tensor log(const Tensor& a);
        static Tensor sqrt(const Tensor& a);
        static Tensor relu(const Tensor& a);
        static Tensor relu_backward(const Tensor& grad_out, const Tensor& a);
        static Tensor pow(const Tensor& a, float exponent);

        static Tensor sum(const Tensor& a, std::optional<size_t> axis = std::nullopt, bool keepdim = false);
        static Tensor mean(const Tensor& a, std::optional<size_t> axis = std::nullopt, bool keepdim = false);

        static Tensor cross_entropy(const Tensor& pred, const Tensor& target);

        // Optimization kernels routing
        static void sgd(Tensor& param, const Tensor& grad, float lr);
    };

}  // namespace helix
