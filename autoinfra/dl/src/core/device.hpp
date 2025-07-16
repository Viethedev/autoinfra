#pragma once

#include <memory>
#include <string>
#include <vector>
#include "core/tensor.hpp"

namespace core {

class Device {
public:
    virtual ~Device() = default;

    // Name of the device, e.g., "cpu:0", "cuda:1"
    virtual std::string name() const = 0;

    // Allocate raw memory on device
    virtual void* alloc(size_t bytes) = 0;

    // Free raw memory
    virtual void free(void* ptr) = 0;

    // Launch a registered kernel
    virtual void run_kernel(const std::string& kernel_name,
                            const std::vector<std::shared_ptr<Tensor>>& inputs,
                            std::vector<std::shared_ptr<Tensor>>& outputs) = 0;

    // Query device properties (num cores, memory, etc.)
    virtual std::string properties() const = 0;
};

using DevicePtr = std::shared_ptr<Device>;

} 
