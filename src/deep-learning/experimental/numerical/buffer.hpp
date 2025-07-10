#pragma once
#include <vector>
#include <memory>
#include <cstddef>
#include <stdexcept>

namespace numerical
{

    /**
     * @brief Abstract base class for data buffers.
     */
    template <typename T>
    class Buffer
    {
    public:
        virtual T *data() = 0;
        virtual const T *data() const = 0;
        virtual ~Buffer() = default;
    };

    /**
     * @brief CPU-resident buffer with ownership.
     */
    template <typename T>
    class CpuBuffer : public Buffer<T>
    {
        std::vector<T> storage_;

    public:
        explicit CpuBuffer(size_t size)
            : storage_(size) {}

        T *data() override { return storage_.data(); }
        const T *data() const override { return storage_.data(); }
    };

    /**
     * @brief A non-owning view into another buffer with offset.
     *
     * Used for slicing without copying data.
     */
    template <typename T>
    class BufferView : public Buffer<T>
    {
        T *data_ptr_;

    public:
        explicit BufferView(T *ptr)
            : data_ptr_(ptr) {}

        T *data() override { return data_ptr_; }
        const T *data() const override { return data_ptr_; }
    };

}
