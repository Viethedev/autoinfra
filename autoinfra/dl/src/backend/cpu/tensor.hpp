#pragma once

#include <vector>
#include <memory>
#include <cstring>
#include "core/tensor.hpp"
#include "core/datatype.hpp"
#include "core/device.hpp"

namespace cpu {

class CpuTensor : public core::Tensor {
public:
    CpuTensor(const std::vector<size_t>& shape, core::DType dtype): shape_(shape), dtype_(dtype) {
        strides_ = compute_default_strides(shape_);
        data_ = std::malloc(nbytes());
    }

    ~CpuTensor() override {
        if (data_) free(data_);
    }
    static std::vector<size_t> compute_contiguous_strides(const std::vector<size_t>& shape) {};

    void to_device(const core::Device& target_device) override {
        if (target_device.type() != core::DeviceType::CPU) {
            throw std::runtime_error("Cross-device copy not implemented yet.");
        }
        // No-op for CPU->CPU
    }

    // Transformations (stubs)
    std::shared_ptr<Tensor> view(const std::vector<size_t>& new_shape) const override {
        // TODO: implement view logic
        throw std::runtime_error("view() not implemented");
    }
    std::shared_ptr<Tensor> reshape(const std::vector<size_t>& new_shape) const override {
        // TODO: implement reshape logic
        throw std::runtime_error("reshape() not implemented");
    }
    std::shared_ptr<Tensor> permute(const std::vector<size_t>& dims) const override {
        throw std::runtime_error("permute() not implemented");
    }
    std::shared_ptr<Tensor> contiguous() const override {
        return std::make_shared<CpuTensor>(*this);
    }
    std::shared_ptr<Tensor> clone() const override {
        auto t = std::make_shared<CpuTensor>(shape_, dtype_, allocator_);
        std::memcpy(t->data(), data_, nbytes());
        return t;
    }

    std::shared_ptr<Tensor> slice(const std::vector<size_t>& start,
                                  const std::vector<size_t>& end,
                                  const std::vector<size_t>& step) const override {
        throw std::runtime_error("slice() not implemented");
    }

    void zero_() override {
        std::memset(data_, 0, nbytes());
    }

    void fill_(double value) override {
        if (dtype_ != core::DType::FLOAT32 && dtype_ != core::DType::FLOAT64) {
            throw std::runtime_error("fill_ only implemented for floating-point");
        }
        size_t n = numel();
        if (dtype_ == core::DType::FLOAT32) {
            float* ptr = static_cast<float*>(data_);
            for (size_t i = 0; i < n; ++i) ptr[i] = static_cast<float>(value);
        } else {
            double* ptr = static_cast<double*>(data_);
            for (size_t i = 0; i < n; ++i) ptr[i] = value;
        }
    }

    void copy_(const core::Tensor& src) override {
        if (src.nbytes() != nbytes()) {
            throw std::runtime_error("copy_: size mismatch");
        }
        std::memcpy(data_, src.data(), nbytes());
    }

private:
    std::vector<size_t> compute_default_strides(const std::vector<size_t>& shape) const {
        std::vector<size_t> strides(shape.size());
        size_t stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    DType dtype_;
    std::shared_ptr<Allocator> allocator_;
    void* data_;
};

} 
