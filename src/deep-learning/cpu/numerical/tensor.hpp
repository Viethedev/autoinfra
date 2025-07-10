#pragma once
#include <cstddef>
#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>

namespace cpu
{
    /**
     * @brief A slicing range similar to Python's slice(start, stop, step)
     */
    class Range
    {
    public:
        size_t start;
        size_t stop;
        size_t step;

        /**
         * @brief Construct a range [start, stop) with step (default = 1)
         */
        Range(size_t start_, size_t stop_, size_t step_ = 1)
            : start(start_), stop(stop_), step(step_) {}
    };

    /**
     * @brief A dynamically-shaped N-dimensional tensor with shared memory,
     *        striding support, and non-owning views for slicing.
     *
     * Features:
     *  - Dynamic rank (N-D tensor)
     *  - Row-major layout
     *  - Shared memory support via shared_ptr
     *  - Non-owning views via slice()
     *  - Basic indexing and slicing support
     *
     * Example usage:
     * @code
     * Tensor<float> t({3, 4});
     * t({1, 2}) = 42.0f;
     *
     * auto view = t.slice({Range(1, 3), Range(0, 4, 2)});
     * std::cout << view({0, 1});  // access sliced view
     * @endcode
     */
    template <typename T>
    class Tensor
    {
    private:
        std::shared_ptr<T[]> buffer_; ///< shared memory buffer
        std::vector<size_t> shape_;   ///< tensor shape (per dimension)
        std::vector<size_t> strides_; ///< row-major strides
        size_t offset_ = 0;           ///< offset in buffer (for views)

    public:
        /*** Constructors ***/

        /**
         * @brief Construct a new tensor with shape (allocates memory)
         * @param shape shape of the tensor (e.g. {3, 4, 5})
         */
        Tensor(const std::vector<size_t> &shape);

        /**
         * @brief Construct a view on existing memory (used internally by slice)
         */
        Tensor(std::shared_ptr<T[]> buffer,
               const std::vector<size_t> &shape,
               const std::vector<size_t> &strides,
               size_t offset);

        /**
         * @brief Default constructor (empty tensor)
         */
        Tensor() = default;

        /*** Access ***/

        /**
         * @brief N-dimensional access via vector of indices
         * @param indices vector of size equal to ndim()
         * @return reference to the element
         */
        T &operator()(const std::vector<size_t> &indices);

        /**
         * @brief Const version of operator()
         */
        const T &operator()(const std::vector<size_t> &indices) const;

        /*** Slicing ***/

        /**
         * @brief Return a view (non-owning) of the tensor using N-dimensional slicing
         * @param ranges one Range per dimension (size must equal ndim())
         * @return a Tensor<T> view (shares memory, no copy)
         */
        Tensor<T> slice(const std::vector<Range> &ranges) const;

        /*** Metadata Accessors ***/

        /**
         * @return number of dimensions (rank)
         */
        size_t ndim() const { return shape_.size(); }

        /**
         * @return shape vector (e.g. {3, 4, 5})
         */
        const std::vector<size_t> &shape() const { return shape_; }

        /**
         * @return strides vector (e.g. for row-major access)
         */
        const std::vector<size_t> &strides() const { return strides_; }

        T *raw_buffer() { return buffer_.get() + offset_; }
        const T *raw_buffer() const { return buffer_.get() + offset_; }

    private:
        /**
         * @brief Compute default row-major strides for a given shape
         */
        static std::vector<size_t> compute_strides(const std::vector<size_t> &shape);
    };

    /*** Implementation below ***/

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