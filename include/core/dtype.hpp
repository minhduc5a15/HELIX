#pragma once

#include <string>

namespace helix {

    enum class DType { Float32, Float64, Int32, Int64 };

    inline size_t dtype_size(const DType dtype) {
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

    inline std::string dtype_name(const DType dtype) {
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

    inline bool is_floating_point(const DType dtype) { return dtype == DType::Float32 || dtype == DType::Float64; }

    inline DType promote_to_float(DType dtype) {
        if (is_floating_point(dtype)) return dtype;
        return DType::Float32;
    }
    inline DType promote_types(DType a, DType b) {
        if (a == b) return a;
        if (a == DType::Float64 || b == DType::Float64) return DType::Float64;
        if (a == DType::Float32 || b == DType::Float32) return DType::Float32;
        if (a == DType::Int64 || b == DType::Int64) return DType::Int64;
        return DType::Int32;
    }

    template <typename T>
    inline DType dtype_of();

    template <>
    inline DType dtype_of<float>() {
        return DType::Float32;
    }
    template <>
    inline DType dtype_of<double>() {
        return DType::Float64;
    }
    template <>
    inline DType dtype_of<int32_t>() {
        return DType::Int32;
    }
    template <>
    inline DType dtype_of<int64_t>() {
        return DType::Int64;
    }

}  // namespace helix

#define HELIX_DISPATCH_ALL_TYPES(TYPE, NAME, ...)                                   \
    switch (TYPE) {                                                                 \
        case ::helix::DType::Float32: {                                             \
            using scalar_t = float;                                                 \
            __VA_ARGS__();                                                          \
            break;                                                                  \
        }                                                                           \
        case ::helix::DType::Float64: {                                             \
            using scalar_t = double;                                                \
            __VA_ARGS__();                                                          \
            break;                                                                  \
        }                                                                           \
        case ::helix::DType::Int32: {                                               \
            using scalar_t = int32_t;                                               \
            __VA_ARGS__();                                                          \
            break;                                                                  \
        }                                                                           \
        case ::helix::DType::Int64: {                                               \
            using scalar_t = int64_t;                                               \
            __VA_ARGS__();                                                          \
            break;                                                                  \
        }                                                                           \
        default:                                                                    \
            throw std::runtime_error(std::string("Unsupported dtype for ") + NAME); \
    }
