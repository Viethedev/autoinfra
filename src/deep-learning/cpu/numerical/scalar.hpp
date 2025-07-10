#pragma once
#include "tensor.hpp"
#include <stdexcept>
#include <omp.h>

namespace cpu
{
    /**
     * Tensor + scalar
     */
    template <typename T>
    Tensor<T> operator+(const Tensor<T> &lhs, T scalar)
    {
        Tensor<T> result(lhs.shape());
        size_t total = 1;
        for (auto s : lhs.shape())
            total *= s;

        T *out = result.raw_buffer();
        const T *in = lhs.raw_buffer();

#pragma omp parallel for
        for (size_t i = 0; i < total; ++i)
        {
            out[i] = in[i] + scalar;
        }
        return result;
    }

    /**
     * scalar + Tensor
     */
    template <typename T>
    Tensor<T> operator+(T scalar, const Tensor<T> &rhs)
    {
        return rhs + scalar;
    }

    /**
     * Tensor - scalar
     */
    template <typename T>
    Tensor<T> operator-(const Tensor<T> &lhs, T scalar)
    {
        Tensor<T> result(lhs.shape());
        size_t total = 1;
        for (auto s : lhs.shape())
            total *= s;

        T *out = result.raw_buffer();
        const T *in = lhs.raw_buffer();

#pragma omp parallel for
        for (size_t i = 0; i < total; ++i)
        {
            out[i] = in[i] - scalar;
        }
        return result;
    }

    /**
     * scalar - Tensor
     */
    template <typename T>
    Tensor<T> operator-(T scalar, const Tensor<T> &rhs)
    {
        Tensor<T> result(rhs.shape());
        size_t total = 1;
        for (auto s : rhs.shape())
            total *= s;

        T *out = result.raw_buffer();
        const T *in = rhs.raw_buffer();

#pragma omp parallel for
        for (size_t i = 0; i < total; ++i)
        {
            out[i] = scalar - in[i];
        }
        return result;
    }

    /**
     * Tensor * scalar
     */
    template <typename T>
    Tensor<T> operator*(const Tensor<T> &lhs, T scalar)
    {
        Tensor<T> result(lhs.shape());
        size_t total = 1;
        for (auto s : lhs.shape())
            total *= s;

        T *out = result.raw_buffer();
        const T *in = lhs.raw_buffer();

#pragma omp parallel for
        for (size_t i = 0; i < total; ++i)
        {
            out[i] = in[i] * scalar;
        }
        return result;
    }

    /**
     * scalar * Tensor
     */
    template <typename T>
    Tensor<T> operator*(T scalar, const Tensor<T> &rhs)
    {
        return rhs * scalar;
    }

    /**
     * Tensor / scalar
     */
    template <typename T>
    Tensor<T> operator/(const Tensor<T> &lhs, T scalar)
    {
        Tensor<T> result(lhs.shape());
        size_t total = 1;
        for (auto s : lhs.shape())
            total *= s;

        T *out = result.raw_buffer();
        const T *in = lhs.raw_buffer();

#pragma omp parallel for
        for (size_t i = 0; i < total; ++i)
        {
            out[i] = in[i] / scalar;
        }
        return result;
    }

    /**
     * scalar / Tensor
     */
    template <typename T>
    Tensor<T> operator/(T scalar, const Tensor<T> &rhs)
    {
        Tensor<T> result(rhs.shape());
        size_t total = 1;
        for (auto s : rhs.shape())
            total *= s;

        T *out = result.raw_buffer();
        const T *in = rhs.raw_buffer();

#pragma omp parallel for
        for (size_t i = 0; i < total; ++i)
        {
            out[i] = scalar / in[i];
        }
        return result;
    }
}