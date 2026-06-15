#pragma once

#include <stdexcept>
#include <vector>

#include "shape.hpp"

namespace helix {

    class Stride {
    public:
        Stride() = default;
        explicit Stride(std::vector<size_t> strides) : strides_(std::move(strides)) {}

        // Computes contiguous Row-major strides
        static Stride compute_contiguous(const Shape& shape) {
            if (shape.empty()) return Stride();
            std::vector<size_t> st(shape.rank(), 1);
            for (int i = static_cast<int>(shape.rank()) - 2; i >= 0; --i) {
                st[i] = st[i + 1] * shape[i + 1];
            }
            return Stride(st);
        }

        size_t rank() const { return strides_.size(); }
        bool empty() const { return strides_.empty(); }

        size_t operator[](size_t index) const { return strides_[index]; }
        size_t& operator[](size_t index) { return strides_[index]; }

        const std::vector<size_t>& vec() const { return strides_; }

        bool operator==(const Stride& other) const { return strides_ == other.strides_; }
        bool operator!=(const Stride& other) const { return strides_ != other.strides_; }

        // Computes linear offset from N-dim indices
        size_t compute_offset(const std::vector<size_t>& indices) const {
            if (indices.size() != strides_.size()) {
                throw std::invalid_argument("Indices rank must match stride rank");
            }
            size_t offset = 0;
            for (size_t i = 0; i < strides_.size(); ++i) {
                offset += indices[i] * strides_[i];
            }
            return offset;
        }

    private:
        std::vector<size_t> strides_;
    };

}  // namespace helix
