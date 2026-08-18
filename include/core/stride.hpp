#pragma once

#include <stdexcept>
#include <vector>

#include "core/math_utils.hpp"
#include "shape.hpp"

namespace helix {

    class Stride {
    public:
        Stride() = default;
        explicit Stride(std::vector<ptrdiff_t> strides) : strides_(std::move(strides)) {}

        // Computes contiguous Row-major strides
        static Stride compute_contiguous(const Shape& shape) {
            if (shape.empty()) return Stride();
            std::vector<ptrdiff_t> st(shape.rank(), 1);
            for (int i = static_cast<int>(shape.rank()) - 2; i >= 0; --i) {
                size_t next_stride;
                if (mul_overflow(static_cast<size_t>(st[i + 1]), shape[i + 1], &next_stride)) {
                    throw std::overflow_error("Stride offset exceeds maximum size_t");
                }
                st[i] = static_cast<ptrdiff_t>(next_stride);
            }
            return Stride(st);
        }

        size_t rank() const { return strides_.size(); }
        bool empty() const { return strides_.empty(); }

        ptrdiff_t operator[](size_t index) const { return strides_[index]; }
        ptrdiff_t& operator[](size_t index) { return strides_[index]; }

        const std::vector<ptrdiff_t>& vec() const { return strides_; }

        bool operator==(const Stride& other) const { return strides_ == other.strides_; }
        bool operator!=(const Stride& other) const { return strides_ != other.strides_; }

        // Computes linear offset from N-dim indices
        ptrdiff_t compute_offset(const std::vector<size_t>& indices) const {
            if (indices.size() != strides_.size()) {
                throw std::invalid_argument("Indices rank must match stride rank");
            }
            ptrdiff_t offset = 0;
            for (size_t i = 0; i < strides_.size(); ++i) {
                offset += indices[i] * strides_[i];
            }
            return offset;
        }

    private:
        std::vector<ptrdiff_t> strides_;
    };

}  // namespace helix
