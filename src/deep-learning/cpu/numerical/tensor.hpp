#pragma once
#include <array>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <iostream>

namespace cpu::numerical
{
    /**
     * @brief N-dimensional fixed-shape Tensor with compile-time shape.
     *
     * Template parameters:
     *   - T: element type
     *   - Shapes...: sizes along each dimension
     *
     * Example:
     *   Tensor<float, 2, 3, 4> t = {1, 2, 3, ..., 24};
     *   float x = t(1, 2, 3);
     */
    template <typename T, size_t... Shapes>
    class Tensor
    {
    public:
        // Number of dimensions (rank)
        static constexpr size_t dims = sizeof...(Shapes);

        // The shape array: sizes along each dimension
        static constexpr std::array<size_t, dims> shape = {Shapes...};

        // Total number of elements
        static constexpr size_t total_size = (Shapes * ...);

    private:
        // Flat storage for elements
        std::array<T, total_size> buffer;

    public:
        /*** Constructors ***/

        // Default
        Tensor() = default;

        // Copy
        Tensor(const Tensor &) = default;
        Tensor &operator=(const Tensor &) = default;

        // Move
        Tensor(Tensor &&) noexcept = default;
        Tensor &operator=(Tensor &&) noexcept = default;

        // Initialize from flat list of elements
        Tensor(std::initializer_list<T> init)
        {
            if (init.size() != total_size)
            {
                throw std::invalid_argument("Initializer list size does not match tensor size");
            }
            std::copy(init.begin(), init.end(), buffer.begin());
        }

        /*** Accessors ***/

        // 1D flat access
        T &operator[](size_t i) { return buffer[i]; }
        const T &operator[](size_t i) const { return buffer[i]; }

        // N-dimensional access
        template <typename... Indices>
        T &operator()(Indices... indices)
        {
            static_assert(sizeof...(Indices) == dims, "Incorrect number of indices");
            return buffer[compute_offset(indices...)];
        }

        template <typename... Indices>
        const T &operator()(Indices... indices) const
        {
            static_assert(sizeof...(Indices) == dims, "Incorrect number of indices");
            return buffer[compute_offset(indices...)];
        }

        // Slicing

    private:
        /*** Helpers ***/

        // Compute flat offset from N-dimensional indices
        template <typename... Indices>
        size_t compute_offset(Indices... indices) const
        {
            std::array<size_t, dims> idx = {static_cast<size_t>(indices)...};
            std::array<size_t, dims> strides = compute_strides();

            size_t offset = 0;
            for (size_t i = 0; i < dims; ++i)
            {
                if (idx[i] >= shape[i])
                {
                    throw std::out_of_range("Index out of bounds");
                }
                offset += idx[i] * strides[i];
            }
            return offset;
        }

        // Compute strides assuming row-major storage
        static constexpr std::array<size_t, dims> compute_strides()
        {
            std::array<size_t, dims> strides{};
            size_t stride = 1;
            for (int i = dims - 1; i >= 0; --i)
            {
                strides[i] = stride;
                stride *= shape[i];
            }
            return strides;
        }
    };
}
