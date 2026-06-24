#pragma once

#include <string>

namespace helix {

    enum class DeviceType { CPU, CUDA };

    struct Device {
        DeviceType type;
        int index;

        Device(const DeviceType type = DeviceType::CPU, const int index = 0) : type(type), index(index) {}

        bool is_cpu() const { return type == DeviceType::CPU; }
        bool is_cuda() const { return type == DeviceType::CUDA; }

        std::string to_string() const {
            if (is_cpu()) return "cpu";
            return "cuda:" + std::to_string(index);
        }

        bool operator==(const Device& other) const { return type == other.type && index == other.index; }
    };

}  // namespace helix
