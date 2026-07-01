#include <algorithm>
#include <cmath>

#include "backend/cpu_backend.hpp"
#include "backend/simd_backend.hpp"

namespace helix {

    void CPUBackend::add(const float* a, const float* b, float* out, const size_t size) {
        if (SIMDBackend::is_supported()) {
            SIMDBackend::add(a, b, out, size);
        } else {
            for (size_t i = 0; i < size; ++i) {
                out[i] = a[i] + b[i];
            }
        }
    }

    void CPUBackend::sub(const float* a, const float* b, float* out, const size_t size) {
        if (SIMDBackend::is_supported()) {
            SIMDBackend::sub(a, b, out, size);
        } else {
            for (size_t i = 0; i < size; ++i) {
                out[i] = a[i] - b[i];
            }
        }
    }

    void CPUBackend::mul(const float* a, const float* b, float* out, const size_t size) {
        if (SIMDBackend::is_supported()) {
            SIMDBackend::mul(a, b, out, size);
        } else {
            for (size_t i = 0; i < size; ++i) {
                out[i] = a[i] * b[i];
            }
        }
    }

    void CPUBackend::div(const float* a, const float* b, float* out, const size_t size) {
        if (SIMDBackend::is_supported()) {
            SIMDBackend::div(a, b, out, size);
        } else {
            for (size_t i = 0; i < size; ++i) {
                out[i] = a[i] / b[i];
            }
        }
    }

    void CPUBackend::add_scalar(const float* a, const float scalar, float* out, const size_t size) {
        if (SIMDBackend::is_supported()) {
            SIMDBackend::add_scalar(a, scalar, out, size);
        } else {
            for (size_t i = 0; i < size; ++i) {
                out[i] = a[i] + scalar;
            }
        }
    }

    void CPUBackend::sub_scalar(const float* a, const float scalar, float* out, const size_t size) {
        if (SIMDBackend::is_supported()) {
            SIMDBackend::sub_scalar(a, scalar, out, size);
        } else {
            for (size_t i = 0; i < size; ++i) {
                out[i] = a[i] - scalar;
            }
        }
    }

    void CPUBackend::mul_scalar(const float* a, const float scalar, float* out, const size_t size) {
        if (SIMDBackend::is_supported()) {
            SIMDBackend::mul_scalar(a, scalar, out, size);
        } else {
            for (size_t i = 0; i < size; ++i) {
                out[i] = a[i] * scalar;
            }
        }
    }

    void CPUBackend::div_scalar(const float* a, const float scalar, float* out, const size_t size) {
        if (SIMDBackend::is_supported()) {
            SIMDBackend::div_scalar(a, scalar, out, size);
        } else {
            for (size_t i = 0; i < size; ++i) {
                out[i] = a[i] / scalar;
            }
        }
    }

    void CPUBackend::neg(const float* a, float* out, const size_t size) {
        if (SIMDBackend::is_supported()) {
            SIMDBackend::neg(a, out, size);
        } else {
            for (size_t i = 0; i < size; ++i) {
                out[i] = -a[i];
            }
        }
    }

    void CPUBackend::exp(const float* a, float* out, const size_t size) {
        for (size_t i = 0; i < size; ++i) {
            out[i] = std::exp(a[i]);
        }
    }

    void CPUBackend::log(const float* a, float* out, const size_t size) {
        for (size_t i = 0; i < size; ++i) {
            out[i] = std::log(a[i]);
        }
    }

    void CPUBackend::sqrt(const float* a, float* out, const size_t size) {
        for (size_t i = 0; i < size; ++i) {
            out[i] = std::sqrt(a[i]);
        }
    }

    void CPUBackend::relu(const float* a, float* out, const size_t size) {
        if (SIMDBackend::is_supported()) {
            SIMDBackend::relu(a, out, size);
        } else {
            for (size_t i = 0; i < size; ++i) {
                out[i] = std::max(0.0f, a[i]);
            }
        }
    }

    void CPUBackend::relu_backward(const float* grad_out, const float* a, float* grad_in, const size_t size) {
        for (size_t i = 0; i < size; ++i) {
            grad_in[i] = a[i] > 0.0f ? grad_out[i] : 0.0f;
        }
    }

    void CPUBackend::pow(const float* a, const float exponent, float* out, const size_t size) {
        for (size_t i = 0; i < size; ++i) {
            out[i] = std::pow(a[i], exponent);
        }
    }

    inline void kernel_sgd(float* param, const float* grad, const float lr, const size_t size) {
        for (size_t i = 0; i < size; ++i) {
            param[i] -= lr * grad[i];
        }
    }

    void CPUBackend::sgd(float* param, const float* grad, const float lr, const size_t size) {
        kernel_sgd(param, grad, lr, size);
    }

}  // namespace helix
