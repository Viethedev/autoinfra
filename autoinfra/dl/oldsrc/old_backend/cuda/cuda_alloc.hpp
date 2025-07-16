#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include "device.hpp"
#include "allocator.hpp"

namespace dl {

/**
 * CUDA Allocator skeleton (placeholder).
 */
class CudaAllocator : public Allocator {
public:
    CudaAllocator(int index = 0) : index_(index) {}

    void* allocate(size_t nbytes) override {
        void* ptr = nullptr;
        // cudaMalloc(&ptr, nbytes);
        return ptr;
    }

    void free(void* ptr) override {
        // cudaFree(ptr);
    }

    Device device() const override {
        return Device(DeviceType::CUDA, index_);
    }

private:
    int index_;
};

} // namespace dl
