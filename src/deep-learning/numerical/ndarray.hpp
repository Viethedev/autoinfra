#pragma once
#include "buffer.hpp"
#include "device.hpp"
#include <memory>
#include <vector>
#include <cassert>
#include <stdexcept>

/*
Device cpu = Device::CPU();
Device cuda_gpu0 = Device::GPU(0, GpuType::CUDA);
Device amd_gpu1  = Device::GPU(1, GpuType::ROCm);

NDArray<float> x_cpu({10, 10}, cpu);
NDArray<float> x_cuda({10, 10}, cuda_gpu0);
NDArray<float> x_rocm({10, 10}, amd_gpu1);
*/

namespace numerical
{
    template <typename T>
    class NDArray
    {
        std::vector<size_t> shape_;
        size_t size_;
        std::unique_ptr<Buffer<T>> buffer_;
        Device device_;

    public:
        NDArray(const std::vector<size_t> &shape, Device device)
            : shape_(shape), size_(compute_size(shape)), device_(device)
        {
            if (device_.is_gpu)
            {
                if (device_.gpu_type == DeviceType::CUDA)
                {
                    buffer_ = std::make_unique<CudaBuffer<T>>(size_, device_.device_id);
                }
                else if (device_.gpu_type == DeviceType::ROCm)
                {
                    buffer_ = std::make_unique<RocmBuffer<T>>(size_, device_.device_id);
                }
                else
                {
                    throw std::runtime_error("Unsupported GPU type!");
                }
            }
            else
            {
                buffer_ = std::make_unique<CpuBuffer<T>>(size_);
            }
        }

        T *data() { return buffer_->data(); }
        const T *data() const { return buffer_->data(); }

        const std::vector<size_t> &shape() const { return shape_; }
        size_t size() const { return size_; }
        Device device() const { return device_; }

        // Flat indexing
        T &operator[](size_t i)
        {
            assert(i < size_);
            return data()[i];
        }

        const T &operator[](size_t i) const
        {
            assert(i < size_);
            return data()[i];
        }

    private:
        static size_t compute_size(const std::vector<size_t> &shape)
        {
            size_t s = 1;
            for (size_t d : shape)
                s *= d;
            return s;
        }
    };

}
