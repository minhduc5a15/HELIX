#include "core/value.hpp"

namespace helix {

    Value::Value(float data) : ptr_(std::make_shared<ValueNode>(data)) {}

    Value::Value(std::shared_ptr<ValueNode> node) : ptr_(std::move(node)) {}

    std::shared_ptr<ValueNode> Value::node() const { return ptr_; }

    float Value::data() const { return ptr_->data; }

    float Value::grad() const { return ptr_->grad; }

    void Value::set_grad(const float g) const { ptr_->grad = g; }

    void Value::add_grad(const float g) const { ptr_->grad += g; }

}  // namespace helix
