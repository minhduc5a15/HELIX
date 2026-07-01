#pragma once

#include <string>
#include <vector>

#include "core/tensor.hpp"

namespace helix {

    /**
     * @class Module
     * @brief Base class for all neural network modules.
     *
     * Your models should also subclass this class.
     * Modules can contain other Modules, allowing for a tree-like structure.
     */
    class Module {
    public:
        virtual ~Module() = default;

        /**
         * @brief Defines the computation performed at every call.
         * @param input The input Tensor.
         * @return The output Tensor.
         */
        virtual Tensor forward(const Tensor& input) = 0;

        /**
         * @brief Overloads the () operator to call forward().
         * @param input The input Tensor.
         * @return The output Tensor.
         */
        Tensor operator()(const Tensor& input) { return forward(input); }

        /**
         * @brief Returns a dictionary-like structure of module parameters.
         * @return A vector of pairs containing the parameter name and the Tensor itself.
         */
        virtual std::vector<std::pair<std::string, Tensor>> named_parameters() { return {}; }

        /**
         * @brief Returns an iterator over module parameters.
         * @return A vector of parameter Tensors.
         */
        virtual std::vector<Tensor> parameters() {
            std::vector<Tensor> params;
            for (const auto& [name, param] : named_parameters()) {
                params.push_back(param);
            }
            return params;
        }
    };

}  // namespace helix
