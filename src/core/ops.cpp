#include <cmath>

#include "core/value.hpp"

namespace helix {

    Value operator+(const Value& a, const Value& b) {
        const auto out_node = std::make_shared<ValueNode>(a.data() + b.data());
        out_node->parents = {a.node(), b.node()};
        out_node->operation = "+";

        auto a_node = a.node();
        auto b_node = b.node();
        ValueNode* out_ptr = out_node.get();  // Raw pointer to avoid cyclic reference

        out_node->backward_fn = [a_node, b_node, out_ptr]() {
            a_node->grad += 1.0f * out_ptr->grad;
            b_node->grad += 1.0f * out_ptr->grad;
        };

        return Value(out_node);
    }

    Value operator*(const Value& a, const Value& b) {
        const auto out_node = std::make_shared<ValueNode>(a.data() * b.data());
        out_node->parents = {a.node(), b.node()};
        out_node->operation = "*";

        auto a_node = a.node();
        auto b_node = b.node();
        ValueNode* out_ptr = out_node.get();

        out_node->backward_fn = [a_node, b_node, out_ptr]() {
            a_node->grad += b_node->data * out_ptr->grad;
            b_node->grad += a_node->data * out_ptr->grad;
        };

        return Value(out_node);
    }

    Value Value::pow(float exponent) const {
        const auto out_node = std::make_shared<ValueNode>(std::pow(data(), exponent));
        out_node->parents = {node()};
        out_node->operation = "pow";

        auto self_node = node();
        ValueNode* out_ptr = out_node.get();

        out_node->backward_fn = [self_node, out_ptr, exponent]() {
            self_node->grad += (exponent * std::pow(self_node->data, exponent - 1.0f)) * out_ptr->grad;
        };

        return Value(out_node);
    }

    Value Value::operator-() const { return *this * Value(-1.0f); }

    Value operator-(const Value& a, const Value& b) { return a + (-b); }

    Value operator/(const Value& a, const Value& b) { return a * b.pow(-1.0f); }

}  // namespace helix
