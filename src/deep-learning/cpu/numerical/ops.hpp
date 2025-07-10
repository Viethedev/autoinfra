#pragma once
#include "base.hpp"
#include <stdexcept>
#include <omp.h>

namespace cpu
{
    /*****************************************************
     * Utility
     *****************************************************/
    inline void assert_same_shape(const std::vector<size_t> &a, const std::vector<size_t> &b)
    {
        if (a != b)
            throw std::invalid_argument("Tensor shapes do not match for element-wise operation");
    }

    /*****************************************************
     *  Scalar Binary Ops
     *    - Tensor + scalar
     *    - scalar + Tensor
     *    - Tensor - scalar
     *    - scalar - Tensor
     *    - Tensor * scalar
     *    - scalar * Tensor
     *    - Tensor / scalar
     *    - scalar / Tensor
     *****************************************************/

    // Tensor + scalar
    template <typename T>
    Tensor<T> operator+(const Tensor<T> &lhs, T scalar)
    {
        Tensor<T> result(lhs.shape());
        size_t total = 1;
        for (auto s : lhs.shape())
            total *= s;

        T *out = result.raw_buffer();
        const T *in = lhs.raw_buffer();

#pragma omp parallel for simd
        for (size_t i = 0; i < total; ++i)
            out[i] = in[i] + scalar;

        return result;
    }

    // scalar + Tensor
    template <typename T>
    Tensor<T> operator+(T scalar, const Tensor<T> &rhs)
    {
        return rhs + scalar;
    }

    // Tensor - scalar
    template <typename T>
    Tensor<T> operator-(const Tensor<T> &lhs, T scalar)
    {
        Tensor<T> result(lhs.shape());
        size_t total = 1;
        for (auto s : lhs.shape())
            total *= s;

        T *out = result.raw_buffer();
        const T *in = lhs.raw_buffer();

#pragma omp parallel for simd
        for (size_t i = 0; i < total; ++i)
            out[i] = in[i] - scalar;

        return result;
    }

    // scalar - Tensor
    template <typename T>
    Tensor<T> operator-(T scalar, const Tensor<T> &rhs)
    {
        Tensor<T> result(rhs.shape());
        size_t total = 1;
        for (auto s : rhs.shape())
            total *= s;

        T *out = result.raw_buffer();
        const T *in = rhs.raw_buffer();

#pragma omp parallel for simd
        for (size_t i = 0; i < total; ++i)
            out[i] = scalar - in[i];

        return result;
    }

    // Tensor * scalar
    template <typename T>
    Tensor<T> operator*(const Tensor<T> &lhs, T scalar)
    {
        Tensor<T> result(lhs.shape());
        size_t total = 1;
        for (auto s : lhs.shape())
            total *= s;

        T *out = result.raw_buffer();
        const T *in = lhs.raw_buffer();

#pragma omp parallel for simd
        for (size_t i = 0; i < total; ++i)
            out[i] = in[i] * scalar;

        return result;
    }

    // scalar * Tensor
    template <typename T>
    Tensor<T> operator*(T scalar, const Tensor<T> &rhs)
    {
        return rhs * scalar;
    }

    // Tensor / scalar
    template <typename T>
    Tensor<T> operator/(const Tensor<T> &lhs, T scalar)
    {
        Tensor<T> result(lhs.shape());
        size_t total = 1;
        for (auto s : lhs.shape())
            total *= s;

        T *out = result.raw_buffer();
        const T *in = lhs.raw_buffer();

#pragma omp parallel for simd
        for (size_t i = 0; i < total; ++i)
            out[i] = in[i] / scalar;

        return result;
    }

    // scalar / Tensor
    template <typename T>
    Tensor<T> operator/(T scalar, const Tensor<T> &rhs)
    {
        Tensor<T> result(rhs.shape());
        size_t total = 1;
        for (auto s : rhs.shape())
            total *= s;

        T *out = result.raw_buffer();
        const T *in = rhs.raw_buffer();

#pragma omp parallel for simd
        for (size_t i = 0; i < total; ++i)
            out[i] = scalar / in[i];

        return result;
    }

    /*****************************************************
     * Element-wise Tensor-Tensor Ops
     *****************************************************/
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

#pragma omp parallel for simd
        for (size_t i = 0; i < total; ++i)
            out[i] = in1[i] + in2[i];

        return result;
    }

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

#pragma omp parallel for simd
        for (size_t i = 0; i < total; ++i)
            out[i] = in1[i] - in2[i];

        return result;
    }

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

#pragma omp parallel for simd
        for (size_t i = 0; i < total; ++i)
            out[i] = in1[i] * in2[i];

        return result;
    }

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

#pragma omp parallel for simd
        for (size_t i = 0; i < total; ++i)
            out[i] = in1[i] / in2[i];

        return result;
    }

    /*****************************************************
     * In-place Ops
     *****************************************************/

    template <typename T>
    Tensor<T> &operator+=(Tensor<T> &lhs, const Tensor<T> &rhs)
    {
        assert_same_shape(lhs.shape(), rhs.shape());
        size_t total = 1;
        for (auto s : lhs.shape())
            total *= s;

        T *ldata = lhs.raw_buffer();
        const T *rdata = rhs.raw_buffer();

#pragma omp parallel for simd
        for (size_t i = 0; i < total; ++i)
            ldata[i] += rdata[i];

        return lhs;
    }

    template <typename T>
    Tensor<T> &operator+=(Tensor<T> &lhs, T scalar)
    {
        size_t total = 1;
        for (auto s : lhs.shape())
            total *= s;

        T *ldata = lhs.raw_buffer();

#pragma omp parallel for simd
        for (size_t i = 0; i < total; ++i)
            ldata[i] += scalar;

        return lhs;
    }

    template <typename T>
    Tensor<T> &operator-=(Tensor<T> &lhs, const Tensor<T> &rhs)
    {
        assert_same_shape(lhs.shape(), rhs.shape());
        size_t total = 1;
        for (auto s : lhs.shape())
            total *= s;

        T *ldata = lhs.raw_buffer();
        const T *rdata = rhs.raw_buffer();

#pragma omp parallel for simd
        for (size_t i = 0; i < total; ++i)
            ldata[i] -= rdata[i];

        return lhs;
    }

    template <typename T>
    Tensor<T> &operator-=(Tensor<T> &lhs, T scalar)
    {
        size_t total = 1;
        for (auto s : lhs.shape())
            total *= s;

        T *ldata = lhs.raw_buffer();

#pragma omp parallel for simd
        for (size_t i = 0; i < total; ++i)
            ldata[i] -= scalar;

        return lhs;
    }

    template <typename T>
    Tensor<T> &operator*=(Tensor<T> &lhs, const Tensor<T> &rhs)
    {
        assert_same_shape(lhs.shape(), rhs.shape());
        size_t total = 1;
        for (auto s : lhs.shape())
            total *= s;

        T *ldata = lhs.raw_buffer();
        const T *rdata = rhs.raw_buffer();

#pragma omp parallel for simd
        for (size_t i = 0; i < total; ++i)
            ldata[i] *= rdata[i];

        return lhs;
    }

    template <typename T>
    Tensor<T> &operator*=(Tensor<T> &lhs, T scalar)
    {
        size_t total = 1;
        for (auto s : lhs.shape())
            total *= s;

        T *ldata = lhs.raw_buffer();

#pragma omp parallel for simd
        for (size_t i = 0; i < total; ++i)
            ldata[i] *= scalar;

        return lhs;
    }

    template <typename T>
    Tensor<T> &operator/=(Tensor<T> &lhs, const Tensor<T> &rhs)
    {
        assert_same_shape(lhs.shape(), rhs.shape());
        size_t total = 1;
        for (auto s : lhs.shape())
            total *= s;

        T *ldata = lhs.raw_buffer();
        const T *rdata = rhs.raw_buffer();

#pragma omp parallel for simd
        for (size_t i = 0; i < total; ++i)
            ldata[i] /= rdata[i];

        return lhs;
    }

    template <typename T>
    Tensor<T> &operator/=(Tensor<T> &lhs, T scalar)
    {
        size_t total = 1;
        for (auto s : lhs.shape())
            total *= s;

        T *ldata = lhs.raw_buffer();

#pragma omp parallel for simd
        for (size_t i = 0; i < total; ++i)
            ldata[i] /= scalar;

        return lhs;
    }

}
