#include <algorithm>
#include <cmath>
#include <type_traits>

#include "backend/cpu_backend.hpp"
#include "backend/simd_backend.hpp"

namespace helix {

    template <typename T>
    void CPUBackend::add(const T* a, const T* b, T* out, const size_t size) {
        if constexpr (std::is_same_v<T, float>) {
            if (SIMDBackend::is_supported()) {
                SIMDBackend::add(a, b, out, size);
                return;
            }
        }
        for (size_t i = 0; i < size; ++i) {
            out[i] = a[i] + b[i];
        }
    }

    template <typename T>
    void CPUBackend::sub(const T* a, const T* b, T* out, const size_t size) {
        if constexpr (std::is_same_v<T, float>) {
            if (SIMDBackend::is_supported()) {
                SIMDBackend::sub(a, b, out, size);
                return;
            }
        }
        for (size_t i = 0; i < size; ++i) {
            out[i] = a[i] - b[i];
        }
    }

    template <typename T>
    void CPUBackend::mul(const T* a, const T* b, T* out, const size_t size) {
        if constexpr (std::is_same_v<T, float>) {
            if (SIMDBackend::is_supported()) {
                SIMDBackend::mul(a, b, out, size);
                return;
            }
        }
        for (size_t i = 0; i < size; ++i) {
            out[i] = a[i] * b[i];
        }
    }

    template <typename T>
    void CPUBackend::div(const T* a, const T* b, T* out, const size_t size) {
        if constexpr (std::is_same_v<T, float>) {
            if (SIMDBackend::is_supported()) {
                SIMDBackend::div(a, b, out, size);
                return;
            }
        }
        for (size_t i = 0; i < size; ++i) {
            out[i] = a[i] / b[i];
        }
    }

    template <typename T>
    void CPUBackend::add_scalar(const T* a, const T scalar, T* out, const size_t size) {
        if constexpr (std::is_same_v<T, float>) {
            if (SIMDBackend::is_supported()) {
                SIMDBackend::add_scalar(a, scalar, out, size);
                return;
            }
        }
        for (size_t i = 0; i < size; ++i) {
            out[i] = a[i] + scalar;
        }
    }

    template <typename T>
    void CPUBackend::sub_scalar(const T* a, const T scalar, T* out, const size_t size) {
        if constexpr (std::is_same_v<T, float>) {
            if (SIMDBackend::is_supported()) {
                SIMDBackend::sub_scalar(a, scalar, out, size);
                return;
            }
        }
        for (size_t i = 0; i < size; ++i) {
            out[i] = a[i] - scalar;
        }
    }

    template <typename T>
    void CPUBackend::mul_scalar(const T* a, const T scalar, T* out, const size_t size) {
        if constexpr (std::is_same_v<T, float>) {
            if (SIMDBackend::is_supported()) {
                SIMDBackend::mul_scalar(a, scalar, out, size);
                return;
            }
        }
        for (size_t i = 0; i < size; ++i) {
            out[i] = a[i] * scalar;
        }
    }

    template <typename T>
    void CPUBackend::div_scalar(const T* a, const T scalar, T* out, const size_t size) {
        if constexpr (std::is_same_v<T, float>) {
            if (SIMDBackend::is_supported()) {
                SIMDBackend::div_scalar(a, scalar, out, size);
                return;
            }
        }
        for (size_t i = 0; i < size; ++i) {
            out[i] = a[i] / scalar;
        }
    }

    template <typename T>
    void CPUBackend::neg(const T* a, T* out, const size_t size) {
        if constexpr (std::is_same_v<T, float>) {
            if (SIMDBackend::is_supported()) {
                SIMDBackend::neg(a, out, size);
                return;
            }
        }
        for (size_t i = 0; i < size; ++i) {
            out[i] = -a[i];
        }
    }

    template <typename T>
    void CPUBackend::exp(const T* a, T* out, const size_t size) {
        for (size_t i = 0; i < size; ++i) {
            out[i] = static_cast<T>(std::exp(static_cast<double>(a[i])));
        }
    }

    template <typename T>
    void CPUBackend::tanh(const T* a, T* out, const size_t size) {
        for (size_t i = 0; i < size; ++i) {
            out[i] = static_cast<T>(std::tanh(static_cast<double>(a[i])));
        }
    }

    template <typename T>
    void CPUBackend::log(const T* a, T* out, const size_t size) {
        for (size_t i = 0; i < size; ++i) {
            out[i] = static_cast<T>(std::log(static_cast<double>(a[i])));
        }
    }

    template <typename T>
    void CPUBackend::sqrt(const T* a, T* out, const size_t size) {
        for (size_t i = 0; i < size; ++i) {
            out[i] = static_cast<T>(std::sqrt(static_cast<double>(a[i])));
        }
    }

    template <typename T>
    void CPUBackend::relu(const T* a, T* out, const size_t size) {
        if constexpr (std::is_same_v<T, float>) {
            if (SIMDBackend::is_supported()) {
                SIMDBackend::relu(a, out, size);
                return;
            }
        }
        for (size_t i = 0; i < size; ++i) {
            out[i] = std::max(static_cast<T>(0), a[i]);
        }
    }

    template <typename T>
    void CPUBackend::relu_backward(const T* grad_out, const T* a, T* grad_in, const size_t size) {
        for (size_t i = 0; i < size; ++i) {
            grad_in[i] = a[i] > static_cast<T>(0) ? grad_out[i] : static_cast<T>(0);
        }
    }

    template <typename T>
    void CPUBackend::pow(const T* a, const T exponent, T* out, const size_t size) {
        for (size_t i = 0; i < size; ++i) {
            out[i] = static_cast<T>(std::pow(static_cast<double>(a[i]), static_cast<double>(exponent)));
        }
    }

    template <typename T>
    void CPUBackend::sgd(T* param, const T* grad, const float lr, const size_t size) {
        for (size_t i = 0; i < size; ++i) {
            param[i] = static_cast<T>(param[i] - static_cast<T>(lr) * grad[i]);
        }
    }

// Explicit instantiations
#define INSTANTIATE_OPS(T)                                                      \
    template void CPUBackend::add<T>(const T*, const T*, T*, size_t);           \
    template void CPUBackend::sub<T>(const T*, const T*, T*, size_t);           \
    template void CPUBackend::mul<T>(const T*, const T*, T*, size_t);           \
    template void CPUBackend::div<T>(const T*, const T*, T*, size_t);           \
    template void CPUBackend::add_scalar<T>(const T*, T, T*, size_t);           \
    template void CPUBackend::sub_scalar<T>(const T*, T, T*, size_t);           \
    template void CPUBackend::mul_scalar<T>(const T*, T, T*, size_t);           \
    template void CPUBackend::div_scalar<T>(const T*, T, T*, size_t);           \
    template void CPUBackend::neg<T>(const T*, T*, size_t);                     \
    template void CPUBackend::exp<T>(const T*, T*, size_t);                     \
    template void CPUBackend::tanh<T>(const T*, T*, size_t);                    \
    template void CPUBackend::log<T>(const T*, T*, size_t);                     \
    template void CPUBackend::sqrt<T>(const T*, T*, size_t);                    \
    template void CPUBackend::relu<T>(const T*, T*, size_t);                    \
    template void CPUBackend::relu_backward<T>(const T*, const T*, T*, size_t); \
    template void CPUBackend::pow<T>(const T*, T, T*, size_t);                  \
    template void CPUBackend::sgd<T>(T*, const T*, float, size_t);

    INSTANTIATE_OPS(float)
    INSTANTIATE_OPS(double)
    INSTANTIATE_OPS(int32_t)
    INSTANTIATE_OPS(int64_t)

}  // namespace helix
