#pragma once

#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include "dtype.hpp"
#include "device.hpp"
#include "allocator.hpp"

namespace dl {

class Tensor {
public:
    virtual ~Tensor() = default;

    // Metadata
    virtual const std::vector<size_t>& shape() const = 0;
    virtual const std::vector<size_t>& strides() const = 0;
    virtual size_t ndim() const = 0;
    virtual size_t numel() const = 0;
    virtual size_t nbytes() const = 0;
    virtual DType dtype() const = 0;
    virtual bool is_contiguous() const = 0;

    // Memory access
    virtual void* data() = 0;
    virtual const void* data() const = 0;

    // Device
    virtual Device device() const = 0;
    virtual void to_device(const Device& target_device) = 0;

    // Allocator
    virtual std::shared_ptr<Allocator> allocator() const = 0;

    // Transformations
    virtual std::shared_ptr<Tensor> view(const std::vector<size_t>& new_shape) const = 0;
    virtual std::shared_ptr<Tensor> reshape(const std::vector<size_t>& new_shape) const = 0;
    virtual std::shared_ptr<Tensor> permute(const std::vector<size_t>& dims) const = 0;
    virtual std::shared_ptr<Tensor> contiguous() const = 0;
    virtual std::shared_ptr<Tensor> clone() const = 0;

    // Indexing and slicing
    virtual std::shared_ptr<Tensor> slice(const std::vector<size_t>& start,
                                          const std::vector<size_t>& end,
                                          const std::vector<size_t>& step) const = 0;

    // In-place ops
    virtual void zero_() = 0;
    virtual void fill_(double value) = 0;
    virtual void copy_(const Tensor& src) = 0;
};

} // namespace dl
