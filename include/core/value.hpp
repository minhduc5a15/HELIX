#pragma once

#include "../autograd/node.hpp"

namespace helix {

// Value class acts as a handle (smart pointer wrapper) to the underlying ValueNode.
// This allows value-semantics in C++ (e.g., Value a = 2.0) while sharing the same
// underlying node in the computational graph, similar to Python references.
class Value {
public:
    // Constructor
    explicit Value(float data);

    // Provide access to underlying node for graph construction
    std::shared_ptr<ValueNode> node() const;

    // Accessors for convenience
    float data() const;
    float grad() const;
    void set_grad(float g) const;
    void add_grad(float g) const;

    // Autograd
    void backward() const;

    // Unary operators
    Value operator-() const;

    // Binary operators (friends to allow symmetry, e.g., 2.0f + a)
    friend Value operator+(const Value& a, const Value& b);
    friend Value operator-(const Value& a, const Value& b);
    friend Value operator*(const Value& a, const Value& b);
    friend Value operator/(const Value& a, const Value& b);

    // Other math operations
    Value pow(float exponent) const;

private:
    // Internal constructor used when creating new nodes from operations
    explicit Value(std::shared_ptr<ValueNode> node);

    std::shared_ptr<ValueNode> ptr_;
};

} // namespace helix
