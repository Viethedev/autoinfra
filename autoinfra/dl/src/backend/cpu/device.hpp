#pragma once
#include "core/device.hpp"
#include "backend/cpu/buffer.hpp"

namespace cpu {

class CpuDevice : public core::Device {
public:
    explicit CpuDevice(int index = 0)
        : index_(index) {}

    core::DeviceType type() const override {
        return core::DeviceType::CPU;
    }

    int index() const override {
        return index_;
    }

    std::string name() const override {
        return "cpu:" + std::to_string(index_);
    }

    core::BufferPtr alloc(size_t num_elements, core::DType dtype) override {
        return make_cpu_buffer(num_elements, dtype);
    }

    void free(core::BufferPtr buffer) override {
        // nothing needed, shared_ptr will manage
    }

private:
    int index_;
};

inline core::DevicePtr make_cpu_device(int index = 0) {
    return std::make_shared<CpuDevice>(index);
}

} // namespace cpu
