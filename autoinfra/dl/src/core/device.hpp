#pragma once
#include "core/buffer.hpp"
#include <string>
#include <memory>

namespace core {

enum class DeviceType {
    CPU,
    CUDA,
    ROCM,
    TPU,
    Unknown
};

inline std::string to_string(DeviceType type) {
    switch (type) {
        case DeviceType::CPU:   return "cpu";
        case DeviceType::CUDA:  return "cuda";
        case DeviceType::ROCM:  return "rocm";
        case DeviceType::TPU:   return "tpu";
        default:                return "unknown";
    }
}

class Device {
public:
    virtual ~Device() = default;

    virtual DeviceType type() const = 0;
    virtual int index() const = 0;
    virtual std::string name() const = 0;

    // Allocate memory buffer on this device
    virtual BufferPtr alloc(size_t num_elements, DType dtype) = 0;

    // (Optional) Freeing buffer manually — usually not needed if shared_ptr is used
    virtual void free(BufferPtr) = 0;
};

using DevicePtr = std::shared_ptr<Device>;

} // namespace core
