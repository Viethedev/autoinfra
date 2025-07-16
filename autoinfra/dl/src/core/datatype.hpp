#pragma once

#include <cstddef>
#include <string>
#include <stdexcept>

namespace core {

enum class DType {
    FLOAT16, FLOAT32, FLOAT64, FLOAT128,
    INT8, INT16, INT32, INT64,
    UINT8, UINT16, UINT32, UINT64,
    BOOL
};

inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::FLOAT16: return 2;
        case DType::FLOAT32: return 4;
        case DType::FLOAT64: return 8;
        case DType::FLOAT128: return 16;
        case DType::INT8: return 1;
        case DType::INT16: return 2;
        case DType::INT32: return 4;
        case DType::INT64: return 8;
        case DType::UINT8: return 1;
        case DType::UINT16: return 2;
        case DType::UINT32: return 4;
        case DType::UINT64: return 8;
        case DType::BOOL: return 1;
        default:
            throw std::invalid_argument("Unknown DType");
    }
}

/**
 * Utility function to get readable name.
 */
inline std::string dtype_name(DType dtype) {
    switch (dtype) {
        case DType::FLOAT16: return "float16";
        case DType::FLOAT32: return "float32";
        case DType::FLOAT64: return "float64";
        case DType::FLOAT128: return "float128";
        case DType::INT8: return "int8";
        case DType::INT16: return "int16";
        case DType::INT32: return "int32";
        case DType::INT64: return "int64";
        case DType::UINT8: return "uint8";
        case DType::UINT16: return "uint16";
        case DType::UINT32: return "uint32";
        case DType::UINT64: return "uint64";
        case DType::BOOL: return "bool";
        default:
            throw std::invalid_argument("Unknown DType");
    }
}

} 