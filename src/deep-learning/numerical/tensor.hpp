#pragma once
#include <vector>
#include <memory>
#include "device.hpp"
#include "buffer.hpp"

namespace numerical {

template<typename T>
class Tensor {
public:
    Tensor(const std::vector<size_t>& shape, Device device = Device::CPU);

    T& operator()(const std::vector<size_t>& indices);
    const T& operator()(const std::vector<size_t>& indices) const;

    size_t size() const { return total_size_; }
    const std::vector<size_t>& shape() const { return shape_; }
    Device device() const { return device_; }

    T* data() { return buffer_->data(); }
    const T* data() const { return buffer_->data(); }

private:
    std::vector<size_t> shape_, strides_;
    size_t total_size_;
    Device device_;
    std::unique_ptr<Buffer<T>> buffer_;

    void compute_strides();
    size_t compute_offset(const std::vector<size_t>& indices) const;
};

}
