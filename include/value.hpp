#pragma once

#include <vector>
#include <functional>
#include <memory>
#include <string>

namespace helix {

// Forward declaration of the internal node structure
struct ValueNode;

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

// The actual node in the computational graph
struct ValueNode {
    float data;
    float grad{0.0f};

    // Using shared_ptr to manage memory automatically and avoid dangling pointers.
    // In a computational graph without loops, shared_ptr is safe.
    std::vector<std::shared_ptr<ValueNode>> parents;
    
    // The backward function to compute gradients for parents
    std::function<void()> backward_fn{[](){}};
    
    // For debugging and visualization
    std::string operation;

    explicit ValueNode(const float d) : data(d) {}
};

} // namespace helix
