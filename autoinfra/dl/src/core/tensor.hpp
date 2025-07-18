#pragma once
#include "core/shape.hpp"
#include "core/device.hpp"
#include "core/buffer.hpp"
#include <memory>
#include <stdexcept>

namespace core {

class Tensor {
public:
    Tensor(Shape shape, DType dtype, DevicePtr device)
        : shape_(std::move(shape)), dtype_(dtype), device_(std::move(device)) {
        if (!device_) throw std::invalid_argument("Device is null");
        strides_ = compute_strides(shape_);
        buffer_ = device_->alloc(numel(shape_), dtype_);
    }

    // Create tensor from existing buffer (e.g. for views)
    Tensor(Shape shape, Strides strides, DType dtype, DevicePtr device, BufferPtr buffer)
        : shape_(std::move(shape)), strides_(std::move(strides)),
          dtype_(dtype), device_(std::move(device)), buffer_(std::move(buffer)) {}

    const Shape& shape() const { return shape_; }
    const Strides& strides() const { return strides_; }
    DType dtype() const { return dtype_; }
    DevicePtr device() const { return device_; }
    BufferPtr buffer() const { return buffer_; }

    void* data() { return buffer_->data(); }
    const void* data() const { return buffer_->data(); }

    size_t ndim() const { return shape_.size(); }
    size_t size() const { return numel(shape_); }
    size_t size_bytes() const { return buffer_->size_bytes(); }

    // Element offset (row-major assumption)
    size_t offset(const std::vector<size_t>& indices) const {
        return compute_offset(shape_, strides_, indices);
    }

    // View (reshape with shared buffer)
    Tensor view(const Shape& new_shape) const {
        if (numel(new_shape) != size())
            throw std::invalid_argument("view(): total size mismatch");
        return Tensor(new_shape, compute_strides(new_shape), dtype_, device_, buffer_);
    }

    // Reshape (alias of view)
    Tensor reshape(const Shape& new_shape) const {
        return view(new_shape);
    }

    // Indexing: returns scalar pointer (can be used with reinterpret_cast)
    void* get_ptr(const std::vector<size_t>& indices) {
        return static_cast<uint8_t*>(data()) + offset(indices) * dtype_size(dtype_);
    }

    const void* get_ptr(const std::vector<size_t>& indices) const {
        return static_cast<const uint8_t*>(data()) + offset(indices) * dtype_size(dtype_);
    }

private:
    Shape shape_;
    Strides strides_;
    DType dtype_;
    DevicePtr device_;
    BufferPtr buffer_;
};

using TensorPtr = std::shared_ptr<Tensor>;

inline TensorPtr make_tensor(const Shape& shape, DType dtype, DevicePtr device) {
    return std::make_shared<Tensor>(shape, dtype, device);
}

} // namespace core
