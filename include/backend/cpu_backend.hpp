#pragma once

#include <cstddef>
#include <cstdint>

namespace helix {

    enum class MatMulStrategy : std::uint8_t { Auto, Naive, Blocked, AVX2, OpenMP };

    class CPUBackend {
    public:
        // Core element-wise operations
        template <typename T>
        static void add(const T* a, const T* b, T* out, size_t size);

        template <typename T>
        static void sub(const T* a, const T* b, T* out, size_t size);

        template <typename T>
        static void mul(const T* a, const T* b, T* out, size_t size);

        template <typename T>
        static void div(const T* a, const T* b, T* out, size_t size);

        // Scalar element-wise operations
        template <typename T>
        static void add_scalar(const T* a, T scalar, T* out, size_t size);

        template <typename T>
        static void sub_scalar(const T* a, T scalar, T* out, size_t size);

        template <typename T>
        static void mul_scalar(const T* a, T scalar, T* out, size_t size);

        template <typename T>
        static void div_scalar(const T* a, T scalar, T* out, size_t size);

        // Unary operations
        template <typename T>
        static void neg(const T* a, T* out, size_t size);

        template <typename T>
        static void exp(const T* a, T* out, size_t size);

        template <typename T>
        static void tanh(const T* a, T* out, size_t size);

        template <typename T>
        static void log(const T* a, T* out, size_t size);

        template <typename T>
        static void sqrt(const T* a, T* out, size_t size);

        template <typename T>
        static void relu(const T* a, T* out, size_t size);

        template <typename T>
        static void relu_backward(const T* grad_out, const T* a, T* grad_in, size_t size);

        template <typename T>
        static void pow(const T* a, T exponent, T* out, size_t size);

        // Matrix Multiplication
        template <typename T>
        static void matmul(
            const T* a, const T* b, T* out, size_t M, size_t K, size_t N, MatMulStrategy strategy = MatMulStrategy::Auto
        );

        // Reduce Operations
        template <typename T>
        static void sum(const T* input, T* output, size_t outer_size, size_t dim_size, size_t inner_size);

        template <typename T>
        static void mean(const T* input, T* output, size_t outer_size, size_t dim_size, size_t inner_size);

        // Loss Operations
        // Computes CrossEntropy Loss with Log-Sum-Exp Trick.
        template <typename T>
        static void cross_entropy(const T* pred, const T* target, T* loss_out, T* log_softmax_out, size_t N, size_t C);

        // SGD Optimization Kernel
        template <typename T>
        static void sgd(T* param, const T* grad, float lr, size_t size);
    };

}  // namespace helix
