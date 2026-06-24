#pragma once

#include <memory>
#include <vector>

#include "core/tensor.hpp"

namespace helix {

    class Node {
    public:
        virtual ~Node() = default;

        // Execute the backward pass for this node
        // Receives gradients corresponding to its outputs and returns gradients for its inputs
        virtual std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) = 0;

        void add_next_edge(std::shared_ptr<Node> node) { next_edges_.push_back(std::move(node)); }

        const std::vector<std::shared_ptr<Node>>& next_edges() const { return next_edges_; }

    protected:
        // The edges pointing to the nodes that this node depends on (parents in the forward graph)
        std::vector<std::shared_ptr<Node>> next_edges_;
    };

}  // namespace helix
