#pragma once

#include <functional>
#include <vector>

namespace helix {

    class Tensor; // Forward declaration

    enum class OpCategory { Unary, Binary, Reduce, View, Matrix };

    enum class OpType { Add, Sub, Mul, Div, Sum, Mean, MatMul, Neg, View, Reshape, Transpose, Flatten, Slice };

    struct OperationContext {
        OpCategory category;
        OpType type;
        Tensor& out;
        std::vector<std::reference_wrapper<const Tensor>> inputs;
        
        // Cấu trúc có thể mở rộng thêm attributes (ví dụ: std::any, enum attributes...)
        // Nhưng tạm thời để đơn giản, có thể ép kiểu context nếu cần ở phía GraphBuilder.
    };

    class GraphBuilderInterface {
    public:
        virtual ~GraphBuilderInterface() = default;
        virtual void build(const OperationContext& ctx) = 0;
    };

}  // namespace helix
