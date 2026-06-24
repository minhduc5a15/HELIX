#include "nn/sequential.hpp"

namespace helix {

    Sequential::Sequential(std::vector<std::shared_ptr<Module>> layers) : layers_(std::move(layers)) {}

    Tensor Sequential::forward(const Tensor& input) {
        Tensor out = input;
        for (const auto& layer : layers_) {
            out = (*layer)(out);
        }
        return out;
    }

    std::vector<std::pair<std::string, Tensor>> Sequential::named_parameters() {
        std::vector<std::pair<std::string, Tensor>> params;
        size_t idx = 0;
        for (const auto& layer : layers_) {
            auto layer_params = layer->named_parameters();
            for (const auto& [name, param] : layer_params) {
                params.emplace_back(std::to_string(idx) + "." + name, param);
            }
            idx++;
        }
        return params;
    }

}  // namespace helix
