#include "nn/sequential.hpp"

#include <cstddef>
#include <stdexcept>

namespace helix {

    Sequential::Sequential(std::vector<std::shared_ptr<Module>> layers) : layers_(std::move(layers)) {
        for (size_t i = 0; i < layers_.size(); ++i) {
            if (!layers_[i]) {
                throw std::invalid_argument("Sequential module at index " + std::to_string(i) + " is null.");
            }
        }
    }

    Tensor Sequential::forward(const Tensor& input) {
        Tensor out = input;
        for (size_t i = 0; i < layers_.size(); ++i) {
            if (!layers_[i]) {
                throw std::runtime_error("Sequential module at index " + std::to_string(i) + " is null.");
            }
            out = (*layers_[i])(out);
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
