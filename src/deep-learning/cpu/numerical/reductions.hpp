#pragma once
#include "base.hpp"
#include <omp.h>
#include <limits>
#include <numeric>

namespace cpu
{

    /*** Total Reductions ***/

    // Sum all elements → scalar
    template <typename T>
    T Tensor<T>::sum() const
    {
        size_t total = 1;
        for (auto s : shape_)
            total *= s;
        const T *data = raw_buffer();
        T result = 0;
#pragma omp parallel for reduction(+ : result)
        for (size_t i = 0; i < total; ++i)
            result += data[i];
        return result;
    }

    // Mean all elements → scalar
    template <typename T>
    T Tensor<T>::mean() const
    {
        return sum() / static_cast<T>(std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>()));
    }

    // Min all elements → scalar
    template <typename T>
    T min(const Tensor<T> &tensor)
    {
        size_t total = 1;
        for (auto s : tensor.shape())
            total *= s;
        if (total == 0)
            throw std::runtime_error("Empty tensor");

        T result = std::numeric_limits<T>::max();
        const T *data = tensor.raw_buffer();

#pragma omp parallel for simd reduction(min : result)
        for (size_t i = 0; i < total; ++i)
        {
            if (data[i] < result)
                result = data[i];
        }
        return result;
    }

    // Max all elements → scalar
    template <typename T>
    T max(const Tensor<T> &tensor)
    {
        size_t total = 1;
        for (auto s : tensor.shape())
            total *= s;
        if (total == 0)
            throw std::runtime_error("Empty tensor");

        T result = std::numeric_limits<T>::lowest();
        const T *data = tensor.raw_buffer();

#pragma omp parallel for simd reduction(max : result)
        for (size_t i = 0; i < total; ++i)
        {
            if (data[i] > result)
                result = data[i];
        }
        return result;
    }

    /*** Partial Reductions (along axis) ***/

    // Sum along one axis → tensor
    template <typename T>
    Tensor<T> Tensor<T>::sum_over_axis(size_t axis) const
    {
        if (axis >= ndim())
            throw std::invalid_argument("Axis out of bounds");

        // New shape = input without axis
        std::vector<size_t> new_shape;
        for (size_t i = 0; i < ndim(); ++i)
            if (i != axis)
                new_shape.push_back(shape_[i]);

        Tensor<T> result(new_shape);

        size_t reduced_dim = shape_[axis];
        size_t outer_size = 1;
        for (size_t i = 0; i < axis; ++i)
            outer_size *= shape_[i];
        size_t inner_size = 1;
        for (size_t i = axis + 1; i < ndim(); ++i)
            inner_size *= shape_[i];

        const T *in_data = raw_buffer();
        T *out_data = result.raw_buffer();

#pragma omp parallel for collapse(2)
        for (size_t outer = 0; outer < outer_size; ++outer)
        {
            for (size_t inner = 0; inner < inner_size; ++inner)
            {
                size_t out_index = outer * inner_size + inner;
                out_data[out_index] = T(0);

                for (size_t r = 0; r < reduced_dim; ++r)
                {
                    size_t in_index =
                        outer * reduced_dim * inner_size + r * inner_size + inner;
                    out_data[out_index] += in_data[in_index];
                }
            }
        }

        return result;
    }

    // Mean along one axis → tensor
    template <typename T>
    Tensor<T> Tensor<T>::mean_over_axis(size_t axis) const
    {
        Tensor<T> summed = sum_over_axis(axis);
        T scale = static_cast<T>(shape_[axis]);
        return summed * (1 / scale);
    }

} // namespace cpu
