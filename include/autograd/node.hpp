#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace helix {

    // The actual node in the computational graph
    struct ValueNode {
        float data;
        float grad{0.0f};

        // Using shared_ptr to manage memory automatically and avoid dangling pointers.
        // In a computational graph without loops, shared_ptr is safe.
        std::vector<std::shared_ptr<ValueNode>> parents;

        // The backward function to compute gradients for parents
        std::function<void()> backward_fn{[]() {}};

        // For debugging and visualization
        std::string operation;

        explicit ValueNode(const float d) : data(d) {}
    };

}  // namespace helix
