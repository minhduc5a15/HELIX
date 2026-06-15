#include <ranges>
#include <unordered_set>

#include "value.hpp"

namespace helix {

    void Value::backward() const {
        // 1. Topological sort using DFS
        std::vector<std::shared_ptr<ValueNode>> topo;
        std::unordered_set<ValueNode*> visited;

        // We use std::function to allow recursive lambda calls
        std::function<void(std::shared_ptr<ValueNode>)> build_topo = [&](const std::shared_ptr<ValueNode>& v) {
            if (!visited.contains(v.get())) {
                visited.insert(v.get());
                for (const auto& parent : v->parents) {
                    build_topo(parent);
                }
                topo.push_back(v);
            }
        };

        // Build the topological order starting from this node
        build_topo(node());

        // 2. Seed gradient for the output node
        set_grad(1.0f);

        // 3. Reverse topological traversal
        // We iterate from the end of the topological sort backwards
        for (const auto& it : std::views::reverse(topo)) {
            it->backward_fn();
        }
    }

}  // namespace helix
