#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/broadcast.hpp"
#include "core/tensor.hpp"

namespace helix {

    constexpr size_t MAX_TENSOR_ITERATOR_DIMS = 8;
    constexpr size_t MAX_TENSOR_ITERATOR_OPERANDS = 8;

    /**
     * @brief Describes a common logical iteration space for tensor operands.
     *
     * Input operands are broadcast to the output shape by adjusting their
     * strides. Offsets returned by this class are relative to each operand's
     * data pointer, so storage offsets remain an implementation detail of Tensor.
     */
    class TensorIterator {
    public:
        TensorIterator(Tensor& output, std::initializer_list<const Tensor*> inputs) : output_(&output) {
            shape_ = output.shape();
            if (shape_.rank() > MAX_TENSOR_ITERATOR_DIMS) {
                throw std::invalid_argument("TensorIterator supports at most 8 dimensions");
            }
            if (inputs.size() + 1 > MAX_TENSOR_ITERATOR_OPERANDS) {
                throw std::invalid_argument("TensorIterator supports at most 8 operands");
            }
            strides_.reserve(inputs.size() + 1);
            strides_.push_back(output.stride());

            for (const Tensor* input : inputs) {
                if (input == nullptr) throw std::invalid_argument("TensorIterator received a null input");
                if (input->device() != output.device()) {
                    throw std::invalid_argument("TensorIterator operands must be on the same device");
                }
                strides_.push_back(compute_broadcast_strides(input->shape(), input->stride(), shape_));
            }

            coalesce_dimensions();
        }

        [[nodiscard]] auto numel() const -> size_t { return shape_.numel(); }
        [[nodiscard]] auto rank() const -> size_t { return shape_.rank(); }
        [[nodiscard]] auto shape() const -> const Shape& { return shape_; }
        [[nodiscard]] auto output() const -> Tensor& { return *output_; }

        [[nodiscard]] auto stride(size_t operand) const -> const Stride& {
            if (operand >= strides_.size()) throw std::out_of_range("TensorIterator operand index out of range");
            return strides_[operand];
        }

        [[nodiscard]] auto is_contiguous() const -> bool {
            const Stride contiguous = Stride::compute_contiguous(shape_);
            for (const Stride& stride : strides_) {
                if (stride != contiguous) return false;
            }
            return true;
        }

        [[nodiscard]] auto offset(size_t operand, size_t flat_index) const -> ptrdiff_t {
            if (operand >= strides_.size()) throw std::out_of_range("TensorIterator operand index out of range");

            ptrdiff_t result = 0;
            size_t remaining = flat_index;
            const Stride& operand_stride = strides_[operand];
            for (size_t dimension = rank(); dimension > 0; --dimension) {
                const size_t axis = dimension - 1;
                const size_t extent = shape_[axis];
                if (extent == 0) return 0;
                const size_t index = remaining % extent;
                remaining /= extent;
                result += static_cast<ptrdiff_t>(index) * operand_stride[axis];
            }
            return result;
        }

        template <typename Function>
        void for_each(Function&& function) const {
            OffsetArray offsets{};
            IndexArray indices{};

            for (size_t flat_index = 0; flat_index < numel(); ++flat_index) {
                function(flat_index, offsets);
                advance(offsets, indices);
            }
        }

        template <typename Function>
        void for_each_parallel(Function&& function, size_t grain_size = 4096) const {
            const size_t total_elements = numel();
            if (grain_size == 0) throw std::invalid_argument("TensorIterator grain size must be positive");
            if (total_elements == 0) return;
            const ptrdiff_t chunk_count = total_elements / grain_size + (total_elements % grain_size != 0);

#pragma omp parallel for schedule(static) if (total_elements > grain_size)
            for (ptrdiff_t chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
                const size_t start = chunk_index * grain_size;
                const size_t end = start + std::min(grain_size, total_elements - start);
                OffsetArray offsets{};
                IndexArray indices{};
                initialize_from_flat(start, offsets, indices);

                for (size_t flat_index = start; flat_index < end; ++flat_index) {
                    function(flat_index, offsets);
                    advance(offsets, indices);
                }
            }
        }

    private:
        using OffsetArray = std::array<ptrdiff_t, MAX_TENSOR_ITERATOR_OPERANDS>;
        using IndexArray = std::array<size_t, MAX_TENSOR_ITERATOR_DIMS>;

        void coalesce_dimensions() {
            for (size_t axis = rank(); axis > 1;) {
                const size_t outer_axis = axis - 2;
                bool can_coalesce = true;
                for (const Stride& stride : strides_) {
                    const ptrdiff_t expected = static_cast<ptrdiff_t>(shape_[axis - 1]) * stride[axis - 1];
                    if (stride[outer_axis] != expected) {
                        can_coalesce = false;
                        break;
                    }
                }

                if (!can_coalesce) {
                    --axis;
                    continue;
                }

                std::vector<size_t> shape_values = shape_.vec();
                shape_values[outer_axis] *= shape_values[axis - 1];
                shape_values.erase(shape_values.begin() + static_cast<ptrdiff_t>(axis - 1));
                shape_ = Shape(std::move(shape_values));
                for (Stride& stride : strides_) {
                    std::vector<ptrdiff_t> values = stride.vec();
                    values[outer_axis] = values[axis - 1];
                    values.erase(values.begin() + static_cast<ptrdiff_t>(axis - 1));
                    stride = Stride(std::move(values));
                }
                --axis;
            }
        }

        void initialize_from_flat(size_t flat_index, OffsetArray& offsets, IndexArray& indices) const {
            size_t remaining = flat_index;
            for (size_t dimension = rank(); dimension > 0; --dimension) {
                const size_t axis = dimension - 1;
                const size_t extent = shape_[axis];
                if (extent == 0) return;
                indices[axis] = remaining % extent;
                remaining /= extent;
            }

            for (size_t operand = 0; operand < strides_.size(); ++operand) {
                offsets[operand] = 0;
                for (size_t axis = 0; axis < rank(); ++axis) {
                    offsets[operand] += static_cast<ptrdiff_t>(indices[axis]) * strides_[operand][axis];
                }
            }
        }

        void advance(OffsetArray& offsets, IndexArray& indices) const {
            for (size_t dimension = rank(); dimension > 0; --dimension) {
                const size_t axis = dimension - 1;
                for (size_t operand = 0; operand < strides_.size(); ++operand) {
                    offsets[operand] += strides_[operand][axis];
                }
                if (++indices[axis] < shape_[axis]) return;
                indices[axis] = 0;
                for (size_t operand = 0; operand < strides_.size(); ++operand) {
                    offsets[operand] -= strides_[operand][axis] * static_cast<ptrdiff_t>(shape_[axis]);
                }
            }
        }

        Tensor* output_;
        Shape shape_;
        std::vector<Stride> strides_;
    };

}  // namespace helix
