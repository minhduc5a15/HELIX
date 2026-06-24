#pragma once

#include <any>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace helix {

    class Tensor;  // Forward declaration

    enum class OpCategory { Unary, Binary, Reduce, View, Matrix };

    enum class OpType {
        Add,
        Sub,
        Mul,
        Div,
        Sum,
        Mean,
        MatMul,
        Neg,
        View,
        Reshape,
        Transpose,
        Flatten,
        Slice,
        Exp,
        Log,
        Sqrt,
        Pow,
        ReLU
    };

    struct OperationContext {
        OpCategory category;
        OpType type;
        Tensor& out;
        std::vector<std::reference_wrapper<const Tensor>> inputs;
        std::unordered_map<std::string, std::any> attributes;
    };

    class GraphBuilderInterface {
    public:
        virtual ~GraphBuilderInterface() = default;
        virtual void build(const OperationContext& ctx) = 0;
    };

}  // namespace helix
