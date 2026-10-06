#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace helix {

    /**
     * @brief Cross-platform multiplication overflow detection for size_t.
     *
     * @param a First operand.
     * @param b Second operand.
     * @param res Pointer to store the result of a * b.
     * @return true if overflow occurred, false otherwise.
     */
    inline bool mul_overflow(size_t a, size_t b, size_t* res) {
#if defined(__GNUC__) || defined(__clang__)
        return __builtin_mul_overflow(a, b, res);
#else
        if (b != 0 && a > std::numeric_limits<size_t>::max() / b) {
            return true;
        }
        *res = a * b;
        return false;
#endif
    }

    /**
     * @brief Cross-platform addition overflow detection for size_t.
     *
     * @param a First operand.
     * @param b Second operand.
     * @param res Pointer to store the result of a + b.
     * @return true if overflow occurred, false otherwise.
     */
    inline bool add_overflow(size_t a, size_t b, size_t* res) {
#if defined(__GNUC__) || defined(__clang__)
        return __builtin_add_overflow(a, b, res);
#else
        if (std::numeric_limits<size_t>::max() - a < b) {
            return true;
        }
        *res = a + b;
        return false;
#endif
    }

    /**
     * @brief Computes stride * extent with overflow detection, handling signed ptrdiff_t strides.
     *
     * @param stride Stride value (can be positive, zero, or negative).
     * @param extent Dimension size (non-negative size_t).
     * @return Result of stride * extent as ptrdiff_t.
     * @throws std::overflow_error if the result exceeds ptrdiff_t limits.
     */
    inline ptrdiff_t checked_stride_mul(ptrdiff_t stride, size_t extent) {
        if (extent == 0 || stride == 0) {
            return 0;
        }
        const size_t abs_stride = (stride < 0) ? static_cast<size_t>(-(stride + 1)) + 1 : static_cast<size_t>(stride);
        size_t product = 0;
        if (mul_overflow(abs_stride, extent, &product) ||
            product > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max())) {
            throw std::overflow_error("Stride multiplication overflowed ptrdiff_t");
        }
        return (stride < 0) ? -static_cast<ptrdiff_t>(product) : static_cast<ptrdiff_t>(product);
    }

}  // namespace helix
