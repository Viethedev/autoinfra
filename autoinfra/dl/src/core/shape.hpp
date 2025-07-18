#pragma once
#include <vector>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <string>

namespace core {

using Shape = std::vector<size_t>;
using Strides = std::vector<size_t>;

// Compute number of elements in a shape
inline size_t numel(const Shape& shape) {
    return std::accumulate(shape.begin(), shape.end(), 1UL, std::multiplies<size_t>());
}

// Compute default strides for a given shape (row-major)
inline Strides compute_strides(const Shape& shape) {
    Strides strides(shape.size());
    size_t stride = 1;
    for (ptrdiff_t i = shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

// Compute linear offset into flat buffer given shape, strides, and indices
inline size_t compute_offset(const Shape& shape, const Strides& strides, const std::vector<size_t>& indices) {
    if (shape.size() != indices.size())
        throw std::invalid_argument("Index rank doesn't match shape rank");

    size_t offset = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (indices[i] >= shape[i])
            throw std::out_of_range("Index out of bounds");
        offset += strides[i] * indices[i];
    }
    return offset;
}

// Convert shape/strides to string for debugging
inline std::string shape_to_string(const Shape& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        oss << shape[i];
        if (i + 1 < shape.size()) oss << ", ";
    }
    oss << "]";
    return oss.str();
}

inline std::string strides_to_string(const Strides& strides) {
    return shape_to_string(strides);
}

} // namespace core
