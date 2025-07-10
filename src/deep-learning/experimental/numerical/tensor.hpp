#pragma once
#include <vector>
#include <memory>
#include <cstddef>
#include <stdexcept>
#include "buffer.hpp"
#include "range.hpp"

namespace numerical
{

    /**
     * @brief N-dimensional Tensor class with buffer abstraction.
     *
     * Supports CPU memory via CpuBuffer. Future GPU buffers possible.
     */
    template <typename T>
    class Tensor
    {
    public:
        /**
         * @brief Construct tensor with given shape. Allocates CPU memory.
         */
        explicit Tensor(const std::vector<size_t> &shape)
            : shape_(shape)
        {
            compute_strides();
            total_size_ = 1;
            for (auto s : shape_)
                total_size_ *= s;

            buffer_ = std::make_unique<CpuBuffer<T> >(total_size_);
        }

        /**
         * @brief Element access by multi-dimensional index.
         *
         * Example:
         *   tensor({0, 2, 1}) = 42;
         */
        T &operator()(const std::vector<size_t> &indices)
        {
            return buffer_->data()[compute_offset(indices)];
        }

        const T &operator()(const std::vector<size_t> &indices) const
        {
            return buffer_->data()[compute_offset(indices)];
        }

        /**
         * @brief Total number of elements.
         */
        size_t size() const { return total_size_; }

        /**
         * @brief Shape of tensor.
         */
        const std::vector<size_t> &shape() const { return shape_; }

        /**
         * @brief Raw data pointer.
         */
        T *data() { return buffer_->data(); }
        const T *data() const { return buffer_->data(); }

        /**
         * @brief Creates a slice of the tensor.
         *
         * Example:
         *   auto t2 = t1.slice(Range(0, 2), Range(1, 4));
         *
         * Slicing is zero-copy via BufferView.
         */
        template <typename... Ranges>
        Tensor slice(const Ranges &...ranges) const
        {
            static_assert(sizeof...(ranges) == shape_.size(), "Number of ranges must match tensor rank");

            // Parse ranges
            std::array<Range, sizeof...(ranges)> range_array{ranges...};

            std::vector<size_t> new_shape(range_array.size());
            std::vector<size_t> new_strides(range_array.size());
            size_t offset = 0;

            for (size_t i = 0; i < range_array.size(); ++i)
            {
                const auto &r = range_array[i];
                if (r.start_ >= shape_[i] || r.stop_ > shape_[i])
                    throw std::out_of_range("Slice range out of bounds");

                new_shape[i] = (r.stop_ - r.start_ + r.step_ - 1) / r.step_;
                new_strides[i] = strides_[i] * r.step_;
                offset += r.start_ * strides_[i];
            }

            size_t new_size = 1;
            for (auto s : new_shape)
                new_size *= s;

            return Tensor(
                new_shape,
                new_strides,
                new_size,
                std::make_unique<BufferView<T> >(buffer_->data() + offset));
        }

    private:
        std::vector<size_t> shape_;
        std::vector<size_t> strides_;
        size_t total_size_;
        std::unique_ptr<Buffer<T> > buffer_;

        /**
         * @brief Compute strides for row-major layout.
         */
        void compute_strides()
        {
            strides_.resize(shape_.size());
            size_t stride = 1;
            for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i)
            {
                strides_[i] = stride;
                stride *= shape_[i];
            }
        }

        /**
         * @brief Compute flat buffer offset from indices.
         */
        size_t compute_offset(const std::vector<size_t> &indices) const
        {
            if (indices.size() != shape_.size())
                throw std::invalid_argument("Incorrect number of indices");

            size_t offset = 0;
            for (size_t i = 0; i < indices.size(); ++i)
            {
                if (indices[i] >= shape_[i])
                    throw std::out_of_range("Index out of bounds");

                offset += indices[i] * strides_[i];
            }
            return offset;
        }

        /**
         * @brief Private constructor for internal use.
         */
        Tensor(
            const std::vector<size_t> &shape,
            const std::vector<size_t> &strides,
            size_t total_size,
            std::unique_ptr<Buffer<T> > &&buffer)
            : shape_(shape),
              strides_(strides),
              total_size_(total_size),
              buffer_(std::move(buffer)) {}
    };

}
