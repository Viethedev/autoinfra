#pragma once

#include "core/device.hpp"
#include <thread>

namespace cpu {

class CpuDevice : public core::Device {
public:
    CpuDevice(int index) : index_(index) {}

    std::string name() const override {
        return "cpu:" + std::to_string(index_);
    }

    void* alloc(size_t bytes) override {
        return std::malloc(bytes);
    }

    void free(void* ptr) override {
        std::free(ptr);
    }

    void run_kernel(const std::string& kernel_name,
                    const std::vector<std::shared_ptr<core::Tensor>>& inputs,
                    std::vector<std::shared_ptr<core::Tensor>>& outputs) override {
        // Look up kernel_name in registry and run it
        // E.g., dispatch_table_[kernel_name](inputs, outputs);
    }

    std::string properties() const override {
        return "CPU device with " + std::to_string(std::thread::hardware_concurrency()) + " threads.";
    }

private:
    int index_;
};

} 
