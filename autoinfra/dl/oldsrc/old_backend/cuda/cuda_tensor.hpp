#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include "tensor.hpp"
#include "allocator.hpp"

namespace dl {

class CudaTensor : public Tensor {
public:
    CudaTensor(const std::vector<size_t>& shape,
               DType dtype,
               std::shared_ptr<Allocator> allocator)
        : shape_(shape), dtype_(dtype), allocator_(allocator) {

        if (!allocator_ || allocator_->device().type() != DeviceType::CUDA) {
            throw std::invalid_argument("CudaTensor requires a CUDA Allocator");
        }

        strides_ = compute_default_strides(shape_);
        data_ = allocator_->allocate(numel() * dtype_size(dtype_));
    }

    ~CudaTensor() override {
        if (data_) {
            allocator_->free(data_);
        }
    }

    // Metadata
    const std::vector<size_t>& shape() const override { return shape_; }
    const std::vector<size_t>& strides() const override { return strides_; }
    size_t ndim() const override { return shape_.size(); }
    size_t numel() const override {
        size_t n = 1;
        for (size_t d : shape_) n *= d;
        return n;
    }
    size_t nbytes() const override {
        return numel() * dtype_size(dtype_);
    }
    DType dtype() const override { return dtype_; }
    bool is_contiguous() const override { return true; }

    // Data access
    void* data() override { return data_; }
    const void* data() const override { return data_; }

    // Device
    Device device() const override { return allocator_->device(); }

    std::shared_ptr<Allocator> allocator() const override { return allocator_; }

    void to_device(const Device& target_device) override {
        throw std::runtime_error("Cross-device copy not implemented yet.");
    }

    // Transformations
    std::shared_ptr<Tensor> view(const std::vector<size_t>& new_shape) const override {
        throw std::runtime_error("view() not implemented");
    }
    std::shared_ptr<Tensor> reshape(const std::vector<size_t>& new_shape) const override {
        throw std::runtime_error("reshape() not implemented");
    }
    std::shared_ptr<Tensor> permute(const std::vector<size_t>& dims) const override {
        throw std::runtime_error("permute() not implemented");
    }
    std::shared_ptr<Tensor> contiguous() const override {
        return std::make_shared<CudaTensor>(*this);
    }
    std::shared_ptr<Tensor> clone() const override {
        auto t = std::make_shared<CudaTensor>(shape_, dtype_, allocator_);
        // TODO: cudaMemcpy here
        return t;
    }

    std::shared_ptr<Tensor> slice(const std::vector<size_t>& start,
                                  const std::vector<size_t>& end,
                                  const std::vector<size_t>& step) const override {
        throw std::runtime_error("slice() not implemented");
    }

    void zero_() override {
        // TODO: cudaMemset
    }

    void fill_(double value) override {
        throw std::runtime_error("fill_ not implemented for CUDA");
    }

    void copy_(const Tensor& src) override {
        throw std::runtime_error("copy_ not implemented for CUDA");
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

} // namespace dl
