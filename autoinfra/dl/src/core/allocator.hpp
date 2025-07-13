#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include "device.hpp"

namespace dl {

/**
 * Abstract Allocator interface.
 */
class Allocator {
public:
    virtual ~Allocator() = default;

    virtual void* allocate(size_t nbytes) = 0;
    virtual void free(void* ptr) = 0;
    virtual Device device() const = 0;
};

/**
 * CPU Allocator implementation.
 */
class CpuAllocator : public Allocator {
public:
    void* allocate(size_t nbytes) override {
        return std::malloc(nbytes);
    }

    void free(void* ptr) override {
        std::free(ptr);
    }

    Device device() const override {
        return Device(DeviceType::CPU);
    }
};

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
