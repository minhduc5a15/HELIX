#pragma once

#include <stdexcept>

#include "core/shape.hpp"
#include "core/stride.hpp"

namespace helix {

    /**
     * @brief Maximum number of dimensions supported by the NDIterator.
     */
    constexpr size_t MAX_DIMS = 8;

    /**
     * @brief NDIterator facilitates efficient iteration over N-dimensional tensors.
     *
     * This struct manages the current N-dimensional index and provides a mechanism
     * to advance through the tensor's elements, updating a linear offset.
     */
    struct NDIterator {
        const Shape& shape;        ///< The shape of the tensor being iterated.
        size_t rank;               ///< The number of dimensions of the tensor.
        size_t indices[MAX_DIMS];  ///< Current N-dimensional indices.

        /**
         * @brief Constructs an NDIterator.
         * @param s The shape of the tensor.
         * @throws std::runtime_error if the tensor's rank exceeds MAX_DIMS.
         */
        explicit NDIterator(const Shape& s) : shape(s), rank(s.rank()) {
            if (rank > MAX_DIMS) {
                throw std::runtime_error("NDIterator: rank exceeds MAX_DIMS (8)");
            }
            for (size_t i = 0; i < MAX_DIMS; ++i) {
                indices[i] = 0;
            }
        }

        /**
         * @brief Initializes the iterator's state from a flat linear index.
         * @param flat_index The logical 1D index to start from.
         */
        inline void init_from_flat(size_t flat_index) {
            size_t current_idx = flat_index;
            for (int j = static_cast<int>(rank) - 1; j >= 0; --j) {
                size_t dim_size = shape[j];
                indices[j] = current_idx % dim_size;
                current_idx /= dim_size;
            }
        }

        /**
         * @brief Computes the memory offset from the current state.
         * @param stride The stride of the tensor.
         * @return The exact memory offset.
         */
        inline size_t compute_offset(const Stride& stride) const {
            size_t offset = 0;
            for (size_t j = 0; j < rank; ++j) {
                offset += indices[j] * stride[j];
            }
            return offset;
        }

        /**
         * @brief Advances the iterator to the next element and updates the linear offset.
         *
         * This method simulates iterating through an N-dimensional array in row-major order,
         * updating the `offset` based on the provided `stride`.
         *
         * @param offset Reference to the linear offset to be updated.
         * @param stride The stride information of the tensor.
         */
        inline void advance(size_t& offset, const Stride& stride) {
            for (int j = static_cast<int>(rank) - 1; j >= 0; --j) {
                indices[j]++;
                offset += stride[j];
                if (indices[j] < shape[j]) {
                    break;
                }
                indices[j] = 0;
                offset -= stride[j] * shape[j];
            }
        }
    };

    /**
     * @brief BinaryNDIterator facilitates efficient simultaneous iteration over two N-dimensional tensors
     * with the same shape.
     *
     * This struct manages the current N-dimensional index and provides a mechanism
     * to advance through the tensors' elements, updating two linear offsets simultaneously.
     */
    struct BinaryNDIterator {
        const Shape& shape;        ///< The shape of the tensors being iterated.
        size_t rank;               ///< The number of dimensions of the tensors.
        size_t indices[MAX_DIMS];  ///< Current N-dimensional indices.

        /**
         * @brief Constructs a BinaryNDIterator.
         * @param s The common shape of the two tensors.
         * @throws std::runtime_error if the tensors' rank exceeds MAX_DIMS.
         */
        explicit BinaryNDIterator(const Shape& s) : shape(s), rank(s.rank()) {
            if (rank > MAX_DIMS) {
                throw std::runtime_error("BinaryNDIterator: rank exceeds MAX_DIMS (8)");
            }
            for (size_t i = 0; i < MAX_DIMS; ++i) {
                indices[i] = 0;
            }
        }

        /**
         * @brief Initializes the iterator's state from a flat linear index.
         * @param flat_index The logical 1D index to start from.
         */
        inline void init_from_flat(size_t flat_index) {
            size_t current_idx = flat_index;
            for (int j = static_cast<int>(rank) - 1; j >= 0; --j) {
                size_t dim_size = shape[j];
                indices[j] = current_idx % dim_size;
                current_idx /= dim_size;
            }
        }

        /**
         * @brief Computes the memory offset from the current state.
         * @param stride The stride of the tensor.
         * @return The exact memory offset.
         */
        inline size_t compute_offset(const Stride& stride) const {
            size_t offset = 0;
            for (size_t j = 0; j < rank; ++j) {
                offset += indices[j] * stride[j];
            }
            return offset;
        }

        /**
         * @brief Advances the iterator for two tensors to the next element and updates their linear offsets.
         *
         * This method simulates iterating through two N-dimensional arrays in row-major order,
         * updating `offset1` and `offset2` based on their respective `stride1` and `stride2`.
         *
         * @param offset1 Reference to the linear offset of the first tensor to be updated.
         * @param stride1 The stride information of the first tensor.
         * @param offset2 Reference to the linear offset of the second tensor to be updated.
         * @param stride2 The stride information of the second tensor.
         */
        inline void advance(size_t& offset1, const Stride& stride1, size_t& offset2, const Stride& stride2) {
            for (int j = static_cast<int>(rank) - 1; j >= 0; --j) {
                indices[j]++;
                offset1 += stride1[j];
                offset2 += stride2[j];
                if (indices[j] < shape[j]) {
                    break;
                }
                indices[j] = 0;
                offset1 -= stride1[j] * shape[j];
                offset2 -= stride2[j] * shape[j];
            }
        }
        /**
         * @brief Computes the memory offset from a flat index without needing state initialization.
         */
        static inline size_t compute_offset_from_flat(size_t flat_index, const Shape& s, const Stride& st) {
            size_t offset = 0;
            size_t current_idx = flat_index;
            for (int j = static_cast<int>(s.rank()) - 1; j >= 0; --j) {
                size_t dim_size = s[j];
                size_t coord = current_idx % dim_size;
                offset += coord * st[j];
                current_idx /= dim_size;
            }
            return offset;
        }
    };

    inline Shape remove_dimension(const Shape& s, size_t dim) {
        std::vector<size_t> dims = s.vec();
        dims.erase(dims.begin() + dim);
        return Shape(dims);
    }

    inline Stride remove_dimension(const Stride& s, size_t dim) {
        std::vector<size_t> strides = s.vec();
        strides.erase(strides.begin() + dim);
        return Stride(strides);
    }
}  // namespace helix
