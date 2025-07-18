#pragma once
#include "core/buffer.hpp"
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace cpu {

// Concrete CPU memory buffer
class CpuBuffer : public core::Buffer {
public:
    CpuBuffer(size_t num_elements, core::DType dtype)
        : dtype_(dtype), size_(num_elements * dtype_size(dtype)) {
        data_ = std::malloc(size_);
        if (!data_) throw std::bad_alloc();
    }

    ~CpuBuffer() override {
        std::free(data_);
    }

    void* data() override {
        return data_;
    }

    const void* data() const override {
        return data_;
    }

    size_t size_bytes() const override {
        return size_;
    }

    core::DType dtype() const override {
        return dtype_;
    }

private:
    void* data_ = nullptr;
    size_t size_ = 0;
    core::DType dtype_;
};

inline core::BufferPtr make_cpu_buffer(size_t num_elements, core::DType dtype) {
    return std::make_shared<CpuBuffer>(num_elements, dtype);
}

} // namespace cpu

