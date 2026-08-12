#pragma once

#include <cstddef>
#include <limits>

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

}  // namespace helix
