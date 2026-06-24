#pragma once

#include "nn/module.hpp"

namespace helix {

    class Linear : public Module {
    public:
        Linear(size_t in_features, size_t out_features);

        Tensor forward(const Tensor& input) override;
        std::vector<std::pair<std::string, Tensor>> named_parameters() override;

    private:
        Tensor weight_;
        Tensor bias_;
    };

}  // namespace helix
