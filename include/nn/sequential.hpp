#pragma once

#include <memory>
#include <type_traits>
#include <vector>

#include "nn/module.hpp"

namespace helix {

    class Sequential : public Module {
    public:
        Sequential() = default;
        Sequential(std::vector<std::shared_ptr<Module>> layers);

        template <typename... Modules>
        Sequential(Modules&&... modules) {
            (layers_.push_back(std::make_shared<std::decay_t<Modules>>(std::forward<Modules>(modules))), ...);
        }

        Tensor forward(const Tensor& input) override;
        std::vector<std::pair<std::string, Tensor>> named_parameters() override;

    private:
        std::vector<std::shared_ptr<Module>> layers_;
    };

}  // namespace helix
