#pragma once
#include <vector>
#include <memory>
#include "tensor.hpp"
#include "range.hpp"

namespace numerical
{
    template <typename T>
    template <typename... Range>
    Tensor<T> Tensor<T>::slice(const Range &...ranges)
    {
        // TODO: parse ranges into new shape and strides
        std::vector<size_t> new_shape(sizeof...(ranges));
        for (i = 0, i < sizeof...(ranges), ++i)
        {
            new_shape[i] = (range[i].stop - range[i].start) / range[i].step
        }
        std::vector<size_t> new_strides = {/* calculated from strides and ranges */};

        size_t new_size = 1;
        for (auto d : new_shape)
            new_size *= d;

        // Optional: compute offset and pass offset pointer into new Buffer wrapper

        // Share the buffer
        return Tensor<T>(new_shape, new_strides, new_size, std::make_unique<BufferView<T>>(data()));
    }

};