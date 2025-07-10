#pragma once
#include <memory>
#include <vector>
#include "range.hpp"

namespace cpu
{

    template <typename T>
    class Tensor
    {
    private:
        std::shared_ptr<T[]> buffer_;
        std::vector<size_t> shape_;
        std::vector<size_t> strides_;
        size_t offset_;

    public:
        Tensor() = default;
        Tensor(const std::vector<size_t> &shape);
        Tensor(std::shared_ptr<T[]> buffer,
               const std::vector<size_t> &shape,
               const std::vector<size_t> &strides,
               size_t offset);

        T &operator()(const std::vector<size_t> &indices);
        const T &operator()(const std::vector<size_t> &indices) const;

        Tensor<T> slice(const std::vector<Range> &ranges) const;

        size_t ndim() const { return shape_.size(); }
        const std::vector<size_t> &shape() const { return shape_; }
        const std::vector<size_t> &strides() const { return strides_; }

        T *raw_buffer() { return buffer_.get() + offset_; }
        const T *raw_buffer() const { return buffer_.get() + offset_; }

        // In-place elementwise
        Tensor<T> &operator+=(const Tensor<T> &);
        Tensor<T> &operator+=(T scalar);
        Tensor<T> &operator-=(const Tensor<T> &);
        Tensor<T> &operator-=(T scalar);
        Tensor<T> &operator*=(const Tensor<T> &);
        Tensor<T> &operator*=(T scalar);
        Tensor<T> &operator/=(const Tensor<T> &);
        Tensor<T> &operator/=(T scalar);

        // Reductions
        T sum() const;
        T mean() const;
        Tensor<T> sum_over_axis(size_t axis) const;
        Tensor<T> mean_over_axis(size_t axis) const;

    private:
        static std::vector<size_t> compute_strides(const std::vector<size_t> &shape);
    };

}
