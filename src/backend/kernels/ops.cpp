#include "backend/cpu_backend.hpp"

namespace helix {

    void CPUBackend::add(const float* a, const float* b, float* out, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            out[i] = a[i] + b[i];
        }
    }

    void CPUBackend::sub(const float* a, const float* b, float* out, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            out[i] = a[i] - b[i];
        }
    }

    void CPUBackend::mul(const float* a, const float* b, float* out, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            out[i] = a[i] * b[i];
        }
    }

    void CPUBackend::div(const float* a, const float* b, float* out, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            out[i] = a[i] / b[i];
        }
    }

    void CPUBackend::neg(const float* a, float* out, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            out[i] = -a[i];
        }
    }

}  // namespace helix
