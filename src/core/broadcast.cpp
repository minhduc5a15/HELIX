#include "core/broadcast.hpp"

#include <algorithm>
#include <stdexcept>

namespace helix {

    Shape compute_broadcast_shape(const Shape& a, const Shape& b) {
        size_t rank_a = a.rank();
        size_t rank_b = b.rank();
        size_t max_rank = std::max(rank_a, rank_b);

        std::vector<size_t> out_dims(max_rank);

        // Iterate right-to-left
        for (size_t i = 0; i < max_rank; ++i) {
            size_t dim_a = (i < rank_a) ? a[rank_a - 1 - i] : 1;
            size_t dim_b = (i < rank_b) ? b[rank_b - 1 - i] : 1;

            if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
                throw std::invalid_argument("Shapes are not broadcastable");
            }

            out_dims[max_rank - 1 - i] = std::max(dim_a, dim_b);
        }

        return Shape(out_dims);
    }

    Stride compute_broadcast_strides(
        const Shape& original_shape, const Stride& original_stride, const Shape& target_shape
    ) {
        if (original_shape.rank() > target_shape.rank()) {
            throw std::invalid_argument("Target shape cannot have smaller rank than original shape");
        }

        size_t rank_orig = original_shape.rank();
        size_t rank_target = target_shape.rank();

        std::vector<size_t> out_strides(rank_target, 0);

        // Iterate right-to-left
        for (size_t i = 0; i < rank_target; ++i) {
            size_t dim_orig = (i < rank_orig) ? original_shape[rank_orig - 1 - i] : 1;
            size_t dim_target = target_shape[rank_target - 1 - i];

            if (dim_orig == dim_target) {
                if (i < rank_orig) {
                    out_strides[rank_target - 1 - i] = original_stride[rank_orig - 1 - i];
                } else {
                    out_strides[rank_target - 1 - i] = 0;
                }
            } else if (dim_orig == 1) {
                out_strides[rank_target - 1 - i] = 0;  // Zero stride for broadcasted dimension
            } else {
                throw std::invalid_argument("Original shape is not broadcastable to target shape");
            }
        }

        return Stride(out_strides);
    }

}  // namespace helix
