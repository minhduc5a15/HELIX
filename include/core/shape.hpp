#pragma once

#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace helix {

    /**
     * @class Shape
     * @brief Represents the shape (dimensions) of a Tensor.
     *
     * This class encapsulates a vector of unsigned integers, where each integer
     * represents the size of a corresponding dimension. It is a core component
     * for defining the structure of a Tensor.
     *
     * Examples:
     * - A scalar has an empty shape `()`.
     * - A vector of 5 elements has shape `(5,)`.
     * - A 3x4 matrix has shape `(3, 4)`.
     */
    class Shape {
    public:
        /**
         * @brief Default constructor. Creates an empty shape (scalar).
         */
        Shape() = default;

        /**
         * @brief Constructor from a std::initializer_list.
         *
         * Allows convenient initialization of a Shape.
         * @code
         * helix::Shape shape = {3, 4, 5};
         * @endcode
         * @param dims A list of dimension sizes.
         */
        Shape(const std::initializer_list<size_t> dims) : dims_(dims) {}

        /**
         * @brief Constructor from a std::vector.
         * @param dims A vector containing the dimension sizes.
         */
        explicit Shape(std::vector<size_t> dims) : dims_(std::move(dims)) {}

        /**
         * @brief Gets the rank (number of dimensions) of the shape.
         * @return The number of dimensions. For example, a (3, 4) shape has a rank of 2.
         */
        size_t rank() const { return dims_.size(); }

        /**
         * @brief Checks if the shape is empty (for a scalar tensor).
         * @return `true` if the shape has no dimensions (rank 0), `false` otherwise.
         */
        bool empty() const { return dims_.empty(); }

        /**
         * @brief Calculates the total number of elements in a tensor with this shape.
         *
         * Computed by multiplying the sizes of all dimensions.
         * For a scalar (empty shape), it returns 1.
         * @return The total number of elements.
         */
        size_t numel() const {
            if (dims_.empty()) return 1;  // Scalar tensor has 1 element
            size_t total = 1;
            for (size_t dim : dims_) {
                if (__builtin_mul_overflow(total, dim, &total)) {
                    throw std::overflow_error("Shape numel exceeds maximum size_t");
                }
            }
            return total;
        }

        /**
         * @brief Accesses the size of a specific dimension (const version).
         * @param index The 0-based index of the dimension to access.
         * @return The size of the dimension at `index`.
         */
        size_t operator[](size_t index) const { return dims_[index]; }

        /**
         * @brief Accesses the size of a specific dimension (modifiable version).
         * @param index The 0-based index of the dimension to access.
         * @return A reference to the size of the dimension at `index`.
         */
        size_t& operator[](size_t index) { return dims_[index]; }

        /**
         * @brief Gets a const reference to the underlying dimension vector.
         * @return A const reference to the vector storing the dimension sizes.
         */
        const std::vector<size_t>& vec() const { return dims_; }

        /**
         * @brief Compares two shapes for equality.
         * @param other The other Shape to compare with.
         * @return `true` if all dimensions are equal, `false` otherwise.
         */
        bool operator==(const Shape& other) const { return dims_ == other.dims_; }

        /**
         * @brief Compares two shapes for inequality.
         * @param other The other Shape to compare with.
         * @return `true` if at least one dimension is different, `false` otherwise.
         */
        bool operator!=(const Shape& other) const { return dims_ != other.dims_; }

        /**
         * @brief Converts the shape to a human-readable string representation.
         *
         * The format is similar to NumPy's.
         * Examples: `()`, `(5,)`, `(3, 4)`.
         * @return A string representing the shape.
         */
        std::string to_string() const {
            if (dims_.empty()) return "()";
            std::string s = "(";
            for (size_t i = 0; i < dims_.size(); ++i) {
                s += std::to_string(dims_[i]);
                if (i < dims_.size() - 1) s += ", ";
            }
            if (dims_.size() == 1) s += ",";  // Convention for 1-D shapes, e.g., (5,)
            s += ")";
            return s;
        }

    private:
        /// @brief Vector storing the size of each dimension.
        std::vector<size_t> dims_;
    };

}  // namespace helix
