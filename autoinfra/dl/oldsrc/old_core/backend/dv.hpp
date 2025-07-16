#pragma once

#include <string>
#include <stdexcept>

namespace dl {

/**
 * DeviceType
 *
 * Types of hardware device backends.
 */
enum class DeviceType {
    CPU,
    CUDA,
    AMDGPU
};

/**
 * Device
 *
 * Identifies a specific device (type + index).
 */
class Device {
public:
    Device(DeviceType type, int index = 0)
        : type_(type), index_(index) {}

    DeviceType type() const { return type_; }
    int index() const { return index_; }

    std::string to_string() const {
        switch (type_) {
            case DeviceType::CPU:
                return "cpu:" + std::to_string(index_);
            case DeviceType::CUDA:
                return "cuda:" + std::to_string(index_);
            case DeviceType::AMDGPU:
                return "amd:" + std::to_string(index_);
            default:
                throw std::invalid_argument("Unknown DeviceType");
        }
    }

    bool operator==(const Device& other) const {
        return type_ == other.type_ && index_ == other.index_;
    }

    bool operator!=(const Device& other) const {
        return !(*this == other);
    }

private:
    DeviceType type_;
    int index_;
};

} // namespace dl
