#pragma once
#include "dtype.hpp"
#include <cstddef>
#include <memory>

namespace core {

// Abstract base buffer class
class Buffer {
public:
    virtual ~Buffer() = default;

    virtual void* data() = 0;
    virtual const void* data() const = 0;
    virtual size_t size_bytes() const = 0;
    virtual DType dtype() const = 0;

    template <typename T>
    T* data_as() {
        return reinterpret_cast<T*>(data());
    }

    template <typename T>
    const T* data_as() const {
        return reinterpret_cast<const T*>(data());
    }
};

using BufferPtr = std::shared_ptr<Buffer>;

} // namespace core
