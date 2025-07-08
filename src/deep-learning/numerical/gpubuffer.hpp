#pragma once
#include <vector>
#include <memory>

/*
Make an abstract GpuBuffer
✅ Extend your polymorphic interface:

template<typename T>
class GpuBuffer : public Buffer<T> {
protected:
    size_t size_;
    int device_id_;
public:
    GpuBuffer(size_t size, int device_id)
        : size_(size), device_id_(device_id) {}
};

*/
namespace numerical
{
    template <typename T>
    class CudaBuffer : public GpuBuffer<T>
    {
        T *device_ptr_ = nullptr;

    public:
        CudaBuffer(size_t size, int device_id)
            : GpuBuffer<T>(size, device_id)
        {
            // cudaSetDevice(device_id);
            // cudaMalloc(&device_ptr_, size * sizeof(T));
        }

        T *data() override { return device_ptr_; }
        const T *data() const override { return device_ptr_; }

        ~CudaBuffer() override
        {
            // cudaFree(device_ptr_);
        }
    };
}