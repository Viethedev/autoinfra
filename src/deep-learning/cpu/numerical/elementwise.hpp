#pragma once
#include "tensor.hpp"
#include <stdexcept>
#include <omp.h>

namespace cpu
{

    /**
     * @brief Check that two shapes match.
     */
    inline void assert_same_shape(const std::vector<size_t> &a, const std::vector<size_t> &b)
    {
        if (a != b)
            throw std::invalid_argument("Tensor shapes do not match for element-wise operation");
    }

    /**
     * @brief Element-wise addition
     */
    template <typename T>
    Tensor<T> operator+(const Tensor<T> &lhs, const Tensor<T> &rhs)
    {
        assert_same_shape(lhs.shape(), rhs.shape());
        Tensor<T> result(lhs.shape());
        size_t total = 1;
        for (auto s : lhs.shape())
            total *= s;

        T *out = result.raw_buffer();
        const T *in1 = lhs.raw_buffer();
        const T *in2 = rhs.raw_buffer();

#pragma omp parallel for
        for (size_t i = 0; i < total; ++i)
        {
            out[i] = in1[i] + in2[i];
        }

        return result;
    }

    /**
     * @brief Element-wise subtraction
     */
    template <typename T>
    Tensor<T> operator-(const Tensor<T> &lhs, const Tensor<T> &rhs)
    {
        assert_same_shape(lhs.shape(), rhs.shape());
        Tensor<T> result(lhs.shape());
        size_t total = 1;
        for (auto s : lhs.shape())
            total *= s;

        T *out = result.raw_buffer();
        const T *in1 = lhs.raw_buffer();
        const T *in2 = rhs.raw_buffer();

#pragma omp parallel for
        for (size_t i = 0; i < total; ++i)
        {
            out[i] = in1[i] - in2[i];
        }

        return result;
    }

    /**
     * @brief Element-wise multiplication
     */
    template <typename T>
    Tensor<T> operator*(const Tensor<T> &lhs, const Tensor<T> &rhs)
    {
        assert_same_shape(lhs.shape(), rhs.shape());
        Tensor<T> result(lhs.shape());
        size_t total = 1;
        for (auto s : lhs.shape())
            total *= s;

        T *out = result.raw_buffer();
        const T *in1 = lhs.raw_buffer();
        const T *in2 = rhs.raw_buffer();

#pragma omp parallel for
        for (size_t i = 0; i < total; ++i)
        {
            out[i] = in1[i] * in2[i];
        }

        return result;
    }

    /**
     * @brief Element-wise division
     */
    template <typename T>
    Tensor<T> operator/(const Tensor<T> &lhs, const Tensor<T> &rhs)
    {
        assert_same_shape(lhs.shape(), rhs.shape());
        Tensor<T> result(lhs.shape());
        size_t total = 1;
        for (auto s : lhs.shape())
            total *= s;

        T *out = result.raw_buffer();
        const T *in1 = lhs.raw_buffer();
        const T *in2 = rhs.raw_buffer();

#pragma omp parallel for
        for (size_t i = 0; i < total; ++i)
        {
            out[i] = in1[i] / in2[i];
        }

        return result;
    }

}
