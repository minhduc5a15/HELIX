#pragma once

#include <initializer_list>
#include <numeric>
#include <string>
#include <vector>

namespace helix {

    class Shape {
    public:
        Shape() = default;
        Shape(std::initializer_list<size_t> dims) : dims_(dims) {}
        explicit Shape(std::vector<size_t> dims) : dims_(std::move(dims)) {}

        size_t rank() const { return dims_.size(); }
        bool empty() const { return dims_.empty(); }  // Rank 0 (Scalar)

        size_t numel() const {
            // Scalar tensor has rank 0 and 1 element
            if (dims_.empty()) return 1;
            return std::accumulate(dims_.begin(), dims_.end(), size_t{1}, std::multiplies<size_t>());
        }

        size_t operator[](size_t index) const { return dims_[index]; }
        size_t& operator[](size_t index) { return dims_[index]; }

        const std::vector<size_t>& vec() const { return dims_; }

        bool operator==(const Shape& other) const { return dims_ == other.dims_; }
        bool operator!=(const Shape& other) const { return dims_ != other.dims_; }

        std::string to_string() const {
            if (dims_.empty()) return "()";
            std::string s = "(";
            for (size_t i = 0; i < dims_.size(); ++i) {
                s += std::to_string(dims_[i]);
                if (i < dims_.size() - 1) s += ", ";
            }
            if (dims_.size() == 1) s += ",";  // e.g. (5,)
            s += ")";
            return s;
        }

    private:
        std::vector<size_t> dims_;
    };

}  // namespace helix
