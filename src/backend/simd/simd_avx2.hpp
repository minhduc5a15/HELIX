#pragma once

#if defined(__AVX2__)
#include <immintrin.h>

#if defined(__FMA__) || (defined(_MSC_VER) && defined(__AVX2__))
#define HELIX_USE_FMA
#endif
#endif

namespace helix {
    namespace simd {

#if defined(__AVX2__)

        // Load 8 contiguous float32 values into a 256-bit register
        inline __m256 load(const float* ptr) { return _mm256_loadu_ps(ptr); }

        // Store 8 float32 values from a 256-bit register to memory
        inline void store(float* ptr, __m256 vec) { _mm256_storeu_ps(ptr, vec); }

        // Set all 8 float32 values in the register to 0.0f
        inline __m256 setzero() { return _mm256_setzero_ps(); }

        // Broadcast a single float32 value to all 8 lanes of the register
        inline __m256 broadcast(float val) { return _mm256_set1_ps(val); }

        // Fused Multiply-Add: a * b + c
        inline __m256 fmadd(__m256 a, __m256 b, __m256 c) {
#if defined(HELIX_USE_FMA)
            return _mm256_fmadd_ps(a, b, c);
#else
            return _mm256_add_ps(c, _mm256_mul_ps(a, b));
#endif
        }

#endif  // __AVX2__

    }  // namespace simd
}  // namespace helix
