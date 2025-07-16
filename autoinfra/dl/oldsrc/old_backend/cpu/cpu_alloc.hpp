#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include "device.hpp"
#include "allocator.hpp"

namespace dl {

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


} // namespace dl
