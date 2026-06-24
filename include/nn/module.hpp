#pragma once

#include <string>
#include <vector>

#include "core/tensor.hpp"

namespace helix {

    class Module {
    public:
        virtual ~Module() = default;

        virtual Tensor forward(const Tensor& input) = 0;

        Tensor operator()(const Tensor& input) { return forward(input); }

        virtual std::vector<std::pair<std::string, Tensor>> named_parameters() { return {}; }

        virtual std::vector<Tensor> parameters() {
            std::vector<Tensor> params;
            for (const auto& [name, param] : named_parameters()) {
                params.push_back(param);
            }
            return params;
        }
    };

}  // namespace helix
