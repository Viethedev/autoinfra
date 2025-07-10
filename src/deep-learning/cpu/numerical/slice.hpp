#pragma once
#include "base.hpp"

namespace cpu
{

    template <typename T>
    Tensor<T> Tensor<T>::slice(const std::vector<Range> &ranges) const
    {
        if (ranges.size() != shape_.size())
            throw std::invalid_argument("Wrong number of slice ranges");

        std::vector<size_t> new_shape;
        std::vector<size_t> new_strides;
        size_t new_offset = offset_;

        for (size_t i = 0; i < ranges.size(); ++i)
        {
            const auto &r = ranges[i];
            if (r.start >= r.stop || r.stop > shape_[i])
                throw std::out_of_range("Invalid slice range");

            size_t dim_size = (r.stop - r.start + r.step - 1) / r.step;
            new_shape.push_back(dim_size);
            new_strides.push_back(strides_[i] * r.step);
            new_offset += strides_[i] * r.start;
        }

        return Tensor<T>(buffer_, new_shape, new_strides, new_offset);
    }

}
