#pragma once

#include "core/backend.hpp"
#include "backend/cpu/device.hpp"
#include <string>

namespace cpu {

class CpuBackend : public core::Backend {
public:
    std::string name() const override {
        return "CPU";
    }

    std::vector<core::DevicePtr> enumerate_devices() override {
        // For CPU, we might return a single device or multiple (NUMA nodes)
        return { std::make_shared<CpuDevice>(0) };
    }

    std::shared_ptr<core::Compiler> create_compiler() override {
        // Return a CPU graph compiler if you have one
        return nullptr;
    }
};

} // namespace dl
