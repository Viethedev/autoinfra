#pragma once
#include "core/backend.hpp"
#include "backend/cpu/device.hpp"

namespace cpu {

class CpuBackend : public core::Backend {
public:
    CpuBackend(int num_devices = 1) {
        for (int i = 0; i < num_devices; ++i) {
            devices_.push_back(make_cpu_device(i));
        }
    }

    std::string name() const override {
        return "cpu";
    }

    const std::vector<core::DevicePtr>& devices() const override {
        return devices_;
    }

private:
    std::vector<core::DevicePtr> devices_;
};

inline core::BackendPtr make_cpu_backend(int num_devices = 1) {
    return std::make_shared<CpuBackend>(num_devices);
}

} // namespace cpu
