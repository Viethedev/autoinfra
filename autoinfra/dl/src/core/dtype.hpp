#pragma once
#include <string>
#include <stdexcept>

namespace core {

enum class DType {
    Float32,
    Float64,
    Int32,
    Int64,
    Bool,
    Unknown
};

inline std::string to_string(DType dtype) {
    switch (dtype) {
        case DType::Float32: return "float32";
        case DType::Float64: return "float64";
        case DType::Int32:   return "int32";
        case DType::Int64:   return "int64";
        case DType::Bool:    return "bool";
        default:             return "unknown";
    }
}

inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return 4;
        case DType::Float64: return 8;
        case DType::Int32:   return 4;
        case DType::Int64:   return 8;
        case DType::Bool:    return 1;
        default: throw std::runtime_error("Unknown DType");
    }
}

template<typename T>
constexpr DType dtype_of();

template<> constexpr DType dtype_of<float>()  { return DType::Float32; }
template<> constexpr DType dtype_of<double>() { return DType::Float64; }
template<> constexpr DType dtype_of<int>()    { return DType::Int32; }
template<> constexpr DType dtype_of<long>()   { return DType::Int64; }
template<> constexpr DType dtype_of<bool>()   { return DType::Bool; }

}  // namespace core
