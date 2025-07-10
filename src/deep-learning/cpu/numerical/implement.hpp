#pragma once
#include "base.hpp"

namespace cpu
{

    template <typename T>
    Tensor<T>::Tensor(const std::vector<size_t> &shape)
        : shape_(shape), offset_(0)
    {
        size_t total = 1;
        for (auto s : shape)
            total *= s;
        buffer_ = std::shared_ptr<T[]>(new T[total]());
        strides_ = compute_strides(shape_);
    }

    template <typename T>
    Tensor<T>::Tensor(std::shared_ptr<T[]> buffer,
                      const std::vector<size_t> &shape,
                      const std::vector<size_t> &strides,
                      size_t offset)
        : buffer_(std::move(buffer)), shape_(shape), strides_(strides), offset_(offset) {}

    template <typename T>
    std::vector<size_t> Tensor<T>::compute_strides(const std::vector<size_t> &shape)
    {
        std::vector<size_t> strides(shape.size());
        size_t stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i)
        {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    template <typename T>
    T &Tensor<T>::operator()(const std::vector<size_t> &indices)
    {
        if (indices.size() != shape_.size())
            throw std::invalid_argument("Incorrect number of indices");

        size_t linear = offset_;
        for (size_t i = 0; i < indices.size(); ++i)
        {
            if (indices[i] >= shape_[i])
                throw std::out_of_range("Index out of bounds");
            linear += indices[i] * strides_[i];
        }
        return buffer_[linear];
    }

    template <typename T>
    const T &Tensor<T>::operator()(const std::vector<size_t> &indices) const
    {
        if (indices.size() != shape_.size())
            throw std::invalid_argument("Incorrect number of indices");

        size_t linear = offset_;
        for (size_t i = 0; i < indices.size(); ++i)
        {
            if (indices[i] >= shape_[i])
                throw std::out_of_range("Index out of bounds");
            linear += indices[i] * strides_[i];
        }
        return buffer_[linear];
    }

}
