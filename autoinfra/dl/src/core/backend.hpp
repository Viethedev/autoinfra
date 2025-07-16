#pragma once
#include "device.hpp"
#include <memory>
#include <vector>
#include <string>

namespace core {

class Compiler;  // Forward declare

class Backend {
public:
    virtual ~Backend() = default;

    // List available devices on this backend
    virtual std::vector<DevicePtr> enumerate_devices() = 0;

    // Name of the backend (e.g., "CPU", "CUDA", "AMD")
    virtual std::string name() const = 0;

    // (Optional) Provide a graph compiler
    virtual std::shared_ptr<Compiler> create_compiler() = 0;
};

using BackendPtr = std::shared_ptr<Backend>;

} // namespace dl
