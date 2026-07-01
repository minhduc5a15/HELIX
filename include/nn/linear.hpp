#pragma once

#include "nn/module.hpp"

namespace helix {

    /**
     * @class Linear
     * @brief Applies a linear transformation to the incoming data: y = xA^T + b.
     */
    class Linear : public Module {
    public:
        /**
         * @brief Constructs a Linear module.
         * @param in_features Size of each input sample.
         * @param out_features Size of each output sample.
         */
        Linear(size_t in_features, size_t out_features);

        /**
         * @brief Performs the linear transformation.
         * @param input The input Tensor of shape (*, in_features).
         * @return The output Tensor of shape (*, out_features).
         */
        Tensor forward(const Tensor& input) override;

        std::vector<std::pair<std::string, Tensor>> named_parameters() override;

    private:
        Tensor weight_;
        Tensor bias_;
    };

}  // namespace helix
