#include "matmul_kernel.hpp"

namespace helix {
    void avx2_dot_matmul(const float* a, const float* b, float* out, const size_t M, const size_t K, const size_t N) {
        avx2_micro_matmul(a, b, out, M, K, N);
    }
}  // namespace helix
