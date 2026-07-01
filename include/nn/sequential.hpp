#pragma once

#include <memory>
#include <type_traits>
#include <vector>

#include "nn/module.hpp"

namespace helix {

    /**
     * @class Sequential
     * @brief A sequential container. Modules will be added to it in the order they are passed in the constructor.
     *
     * Alternatively, an ordered vector of modules can be passed in.
     */
    class Sequential : public Module {
    public:
        /**
         * @brief Constructs an empty Sequential container.
         */
        Sequential() = default;

        /**
         * @brief Constructs a Sequential container with a vector of layers.
         * @param layers A vector of shared pointers to Module.
         */
        Sequential(std::vector<std::shared_ptr<Module>> layers);

        /**
         * @brief Variadic template constructor to initialize with multiple modules.
         * @param modules A parameter pack of modules.
         */
        template <typename... Modules>
        Sequential(Modules&&... modules) {
            (layers_.push_back(std::make_shared<std::decay_t<Modules>>(std::forward<Modules>(modules))), ...);
        }

        /**
         * @brief Passes the input sequentially through all registered modules.
         * @param input The input Tensor.
         * @return The output Tensor after passing through all modules.
         */
        Tensor forward(const Tensor& input) override;

        std::vector<std::pair<std::string, Tensor>> named_parameters() override;

    private:
        std::vector<std::shared_ptr<Module>> layers_;
    };

}  // namespace helix
