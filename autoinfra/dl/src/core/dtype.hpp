#pragma once

#include <cstddef>
#include <string>
#include <stdexcept>

namespace dl {

/**
 * DType enum
 *
 * Represents scalar element types.
 */
enum class DType {
    FLOAT32,
    FLOAT64,
    INT32,
    INT64,
    BOOL
};

/**
 * Utility function to get size in bytes.
 */
inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::FLOAT32: return 4;
        case DType::FLOAT64: return 8;
        case DType::INT32: return 4;
        case DType::INT64: return 8;
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
        case DType::FLOAT32: return "float32";
        case DType::FLOAT64: return "float64";
        case DType::INT32: return "int32";
        case DType::INT64: return "int64";
        case DType::BOOL: return "bool";
        default:
            throw std::invalid_argument("Unknown DType");
    }
}

} // namespace dl
