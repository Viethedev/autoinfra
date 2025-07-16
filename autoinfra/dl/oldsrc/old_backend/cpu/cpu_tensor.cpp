#pragma once

#include "cpu_tensor.hpp"
#include "cpu_alloc.hpp"
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace dl {

// Constructor
CpuTensor::CpuTensor(
    const std::vector<size_t>& shape,
    DType dtype,
    std::shared_ptr<Allocator> allocator
)
    : shape_(shape),
      dtype_(dtype),
      allocator_(std::move(allocator))
{
    if (!allocator_) {
        throw std::invalid_argument("Allocator cannot be null");
    }
    strides_ = compute_contiguous_strides(shape_);
    data_ = allocator_->allocate(dtype_size(dtype_) * numel());
}

// Destructor
CpuTensor::~CpuTensor() {
    allocator_->free(data_);
}

// Compute contiguous strides
std::vector<size_t> CpuTensor::compute_contiguous_strides(const std::vector<size_t>& shape) {
    std::vector<size_t> strides(shape.size());
    size_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

// Metadata
const std::vector<size_t>& CpuTensor::shape() const { return shape_; }
const std::vector<size_t>& CpuTensor::strides() const { return strides_; }
size_t CpuTensor::ndim() const { return shape_.size(); }
size_t CpuTensor::numel() const {
    size_t n = 1;
    for (auto s : shape_) n *= s;
    return n;
}
size_t CpuTensor::nbytes() const {
    return numel() * dtype_size(dtype_);
}
DType CpuTensor::dtype() const { return dtype_; }
Device CpuTensor::device() const { return allocator_->device(); }
bool CpuTensor::is_contiguous() const {
    return strides_ == compute_contiguous_strides(shape_);
}

// Memory access
void* CpuTensor::data() { return data_; }
const void* CpuTensor::data() const { return data_; }

// In-place ops
void CpuTensor::zero_() {
    std::memset(data_, 0, nbytes());
}

void CpuTensor::fill_(double value) {
    if (dtype_ == DType::FLOAT32) {
        float* ptr = static_cast<float*>(data_);
        std::fill(ptr, ptr + numel(), static_cast<float>(value));
    } else if (dtype_ == DType::FLOAT64) {
        double* ptr = static_cast<double*>(data_);
        std::fill(ptr, ptr + numel(), value);
    } else {
        throw std::runtime_error("Unsupported dtype in fill_()");
    }
}

void CpuTensor::copy_(const Tensor& src) {
    if (src.dtype() != dtype() || src.numel() != numel()) {
        throw std::invalid_argument("Copy: dtype or shape mismatch");
    }
    std::memcpy(data_, src.data(), nbytes());
}

// Transformations
std::shared_ptr<Tensor> CpuTensor::view(const std::vector<size_t>& new_shape) const {
    size_t new_numel = 1;
    for (auto s : new_shape) new_numel *= s;
    if (new_numel != numel()) {
        throw std::invalid_argument("View: new shape has different numel");
    }
    auto result = std::make_shared<CpuTensor>(*this);
    result->shape_ = new_shape;
    result->strides_ = compute_contiguous_strides(new_shape);
    return result;
}

std::shared_ptr<Tensor> CpuTensor::reshape(const std::vector<size_t>& new_shape) const {
    return view(new_shape);
}

std::shared_ptr<Tensor> CpuTensor::permute(const std::vector<size_t>& dims) const {
    if (dims.size() != shape_.size()) {
        throw std::invalid_argument("Permute: dims must match rank");
    }
    auto result = std::make_shared<CpuTensor>(*this);
    std::vector<size_t> new_shape(shape_.size());
    std::vector<size_t> new_strides(shape_.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] >= shape_.size()) {
            throw std::invalid_argument("Permute: invalid dimension index");
        }
        new_shape[i] = shape_[dims[i]];
        new_strides[i] = strides_[dims[i]];
    }
    result->shape_ = new_shape;
    result->strides_ = new_strides;
    return result;
}

std::shared_ptr<Tensor> CpuTensor::contiguous() const {
    if (is_contiguous()) {
        return std::make_shared<CpuTensor>(*this);
    }
    auto result = std::make_shared<CpuTensor>(shape_, dtype_, allocator_);
    // Copy with simplified memcpy (not real strided copy)
    std::memcpy(result->data(), data_, nbytes());
    return result;
}

std::shared_ptr<Tensor> CpuTensor::clone() const {
    auto result = std::make_shared<CpuTensor>(shape_, dtype_, allocator_);
    std::memcpy(result->data(), data_, nbytes());
    return result;
}

std::shared_ptr<Tensor> CpuTensor::slice(
    const std::vector<size_t>& start,
    const std::vector<size_t>& end,
    const std::vector<size_t>& step
) const {
    throw std::runtime_error("Slice() not implemented yet!");
}

} // namespace dl
