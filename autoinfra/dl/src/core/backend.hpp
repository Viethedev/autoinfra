#pragma once
#include "core/device.hpp"
#include <vector>
#include <memory>
#include <string>

namespace core {

class Backend {
public:
    virtual ~Backend() = default;

    // Backend name: e.g. "cpu", "cuda"
    virtual std::string name() const = 0;

    // Devices managed by this backend
    virtual const std::vector<DevicePtr>& devices() const = 0;

    // Future: register and dispatch kernels here
    // virtual KernelRegistry& kernels() = 0;
};

using BackendPtr = std::shared_ptr<Backend>;

} // namespace core
