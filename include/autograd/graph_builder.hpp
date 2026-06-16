#pragma once

#include "core/graph_builder.hpp"

namespace helix {

    class AutogradGraphBuilder : public GraphBuilderInterface {
    public:
        AutogradGraphBuilder() = default;

        // Intercepts forward operations and builds the computational graph
        void build(const OperationContext& ctx) override;
    };

}  // namespace helix
