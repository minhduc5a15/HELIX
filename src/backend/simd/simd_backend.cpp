#include "backend/simd_backend.hpp"

#include <algorithm>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace helix {

    bool SIMDBackend::is_supported() {
#if defined(__AVX2__)
        return true;
#else
        return false;
#endif
    }

    void SIMDBackend::add(const float* a, const float* b, float* out, size_t size) {
        size_t i = 0;
#if defined(__AVX2__)
        for (; i + 8 <= size; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vout = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(out + i, vout);
        }
#endif
        for (; i < size; ++i) out[i] = a[i] + b[i];
    }

    void SIMDBackend::sub(const float* a, const float* b, float* out, size_t size) {
        size_t i = 0;
#if defined(__AVX2__)
        for (; i + 8 <= size; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vout = _mm256_sub_ps(va, vb);
            _mm256_storeu_ps(out + i, vout);
        }
#endif
        for (; i < size; ++i) out[i] = a[i] - b[i];
    }

    void SIMDBackend::mul(const float* a, const float* b, float* out, size_t size) {
        size_t i = 0;
#if defined(__AVX2__)
        for (; i + 8 <= size; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vout = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(out + i, vout);
        }
#endif
        for (; i < size; ++i) out[i] = a[i] * b[i];
    }

    void SIMDBackend::div(const float* a, const float* b, float* out, size_t size) {
        size_t i = 0;
#if defined(__AVX2__)
        for (; i + 8 <= size; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vout = _mm256_div_ps(va, vb);
            _mm256_storeu_ps(out + i, vout);
        }
#endif
        for (; i < size; ++i) out[i] = a[i] / b[i];
    }

    void SIMDBackend::add_scalar(const float* a, float scalar, float* out, size_t size) {
        size_t i = 0;
#if defined(__AVX2__)
        __m256 vscalar = _mm256_set1_ps(scalar);
        for (; i + 8 <= size; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vout = _mm256_add_ps(va, vscalar);
            _mm256_storeu_ps(out + i, vout);
        }
#endif
        for (; i < size; ++i) out[i] = a[i] + scalar;
    }

    void SIMDBackend::sub_scalar(const float* a, float scalar, float* out, size_t size) {
        size_t i = 0;
#if defined(__AVX2__)
        __m256 vscalar = _mm256_set1_ps(scalar);
        for (; i + 8 <= size; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vout = _mm256_sub_ps(va, vscalar);
            _mm256_storeu_ps(out + i, vout);
        }
#endif
        for (; i < size; ++i) out[i] = a[i] - scalar;
    }

    void SIMDBackend::mul_scalar(const float* a, float scalar, float* out, size_t size) {
        size_t i = 0;
#if defined(__AVX2__)
        __m256 vscalar = _mm256_set1_ps(scalar);
        for (; i + 8 <= size; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vout = _mm256_mul_ps(va, vscalar);
            _mm256_storeu_ps(out + i, vout);
        }
#endif
        for (; i < size; ++i) out[i] = a[i] * scalar;
    }

    void SIMDBackend::div_scalar(const float* a, float scalar, float* out, size_t size) {
        size_t i = 0;
#if defined(__AVX2__)
        __m256 vscalar = _mm256_set1_ps(scalar);
        for (; i + 8 <= size; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vout = _mm256_div_ps(va, vscalar);
            _mm256_storeu_ps(out + i, vout);
        }
#endif
        for (; i < size; ++i) out[i] = a[i] / scalar;
    }

    void SIMDBackend::neg(const float* a, float* out, size_t size) {
        size_t i = 0;
#if defined(__AVX2__)
        __m256 vzero = _mm256_setzero_ps();
        for (; i + 8 <= size; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vout = _mm256_sub_ps(vzero, va);
            _mm256_storeu_ps(out + i, vout);
        }
#endif
        for (; i < size; ++i) out[i] = -a[i];
    }

    void SIMDBackend::relu(const float* a, float* out, size_t size) {
        size_t i = 0;
#if defined(__AVX2__)
        __m256 vzero = _mm256_setzero_ps();
        for (; i + 8 <= size; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vout = _mm256_max_ps(vzero, va);
            _mm256_storeu_ps(out + i, vout);
        }
#endif
        for (; i < size; ++i) out[i] = std::max(0.0f, a[i]);
    }

    void SIMDBackend::sum(const float* input, float* output, size_t outer_size, size_t dim_size, size_t inner_size) {
        if (inner_size == 1) {
            for (size_t i = 0; i < outer_size; ++i) {
                const float* ptr = input + i * dim_size;
                float sum_val = 0.0f;
                size_t j = 0;
#if defined(__AVX2__)
                __m256 vsum = _mm256_setzero_ps();
                for (; j + 8 <= dim_size; j += 8) {
                    vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(ptr + j));
                }
                __m128 vlow  = _mm256_castps256_ps128(vsum);
                __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
                vlow  = _mm_add_ps(vlow, vhigh);
                __m128 shuf = _mm_movehl_ps(vlow, vlow);
                vlow  = _mm_add_ps(vlow, shuf);
                shuf  = _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(1, 1, 1, 1));
                vlow  = _mm_add_ss(vlow, shuf);
                sum_val += _mm_cvtss_f32(vlow);
#endif
                for (; j < dim_size; ++j) {
                    sum_val += ptr[j];
                }
                output[i] = sum_val;
            }
        } else {
            // Fallback
            for (size_t i = 0; i < outer_size; ++i) {
                for (size_t k = 0; k < inner_size; ++k) {
                    float sum_val = 0.0f;
                    for (size_t j = 0; j < dim_size; ++j) {
                        sum_val += input[i * (dim_size * inner_size) + j * inner_size + k];
                    }
                    output[i * inner_size + k] = sum_val;
                }
            }
        }
    }

    void SIMDBackend::mean(const float* input, float* output, size_t outer_size, size_t dim_size, size_t inner_size) {
        const float scale = 1.0f / static_cast<float>(dim_size);
        if (inner_size == 1) {
            for (size_t i = 0; i < outer_size; ++i) {
                const float* ptr = input + i * dim_size;
                float sum_val = 0.0f;
                size_t j = 0;
#if defined(__AVX2__)
                __m256 vsum = _mm256_setzero_ps();
                for (; j + 8 <= dim_size; j += 8) {
                    vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(ptr + j));
                }
                __m128 vlow  = _mm256_castps256_ps128(vsum);
                __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
                vlow  = _mm_add_ps(vlow, vhigh);
                __m128 shuf = _mm_movehl_ps(vlow, vlow);
                vlow  = _mm_add_ps(vlow, shuf);
                shuf  = _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(1, 1, 1, 1));
                vlow  = _mm_add_ss(vlow, shuf);
                sum_val += _mm_cvtss_f32(vlow);
#endif
                for (; j < dim_size; ++j) {
                    sum_val += ptr[j];
                }
                output[i] = sum_val * scale;
            }
        } else {
            // Fallback
            for (size_t i = 0; i < outer_size; ++i) {
                for (size_t k = 0; k < inner_size; ++k) {
                    float sum_val = 0.0f;
                    for (size_t j = 0; j < dim_size; ++j) {
                        sum_val += input[i * (dim_size * inner_size) + j * inner_size + k];
                    }
                    output[i * inner_size + k] = sum_val * scale;
                }
            }
        }
    }

}  // namespace helix
