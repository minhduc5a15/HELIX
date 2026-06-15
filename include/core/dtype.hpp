#pragma once

#include <cstdint>
#include <string>

namespace helix {

    enum class DType { Float32, Float64, Int32, Int64 };

    inline size_t dtype_size(DType dtype) {
        switch (dtype) {
            case DType::Float32:
                return 4;
            case DType::Float64:
                return 8;
            case DType::Int32:
                return 4;
            case DType::Int64:
                return 8;
            default:
                return 0;
        }
    }

    constexpr std::string dtype_name(DType dtype) {
        switch (dtype) {
            case DType::Float32:
                return "float32";
            case DType::Float64:
                return "float64";
            case DType::Int32:
                return "int32";
            case DType::Int64:
                return "int64";
            default:
                return "unknown";
        }
    }

}  // namespace helix
