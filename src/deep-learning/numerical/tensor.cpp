#include "tensor.hpp"

namespace numerical
{

    // Constructor
    template <typename T>
    Tensor<T>::Tensor(const std::vector<size_t> &shape) //, Device device)
        : shape_(shape)                                 //, device_(device)
    {
        /*
        if (device_.is_gpu)
        {
            if (device_.device_type == DeviceType::CUDA)
            {
                buffer_ = std::make_unique<CudaBuffer<T>>(total_size_, device_.device_id);
            }
            else if (device_.device_type == DeviceType::ROCm)
            {
                buffer_ = std::make_unique<RocmBuffer<T>>(total_size_, device_.device_id);
            }
            else if (device_.device_type == DeviceType::Vulkan)
            {
                buffer_ = std::make_unique<VulkanBuffer<T>>(total_size_, device_.device_id);
            }
            else if (device_.device_type == DeviceType::Metal)
            {
                buffer_ = std::make_unique<MetalBuffer<T>>(total_size_, device_.device_id);
            }
            else
            {
                throw std::runtime_error("Unsupported GPU type!");
            }
        }
        else
        {
            buffer_ = std::make_unique<CpuBuffer<T>>(total_size_);
        }
        */

        // Deploy buffer
        buffer_ = std::make_unique<CpuBuffer<T>>(total_size_);

        // Compute strides
        compute_strides();

        // Compute total size
        total_size_ = 1;
        for (auto s : shape_)
            total_size_ *= s;
    }

    // Compute strides method
    template <typename T>
    void Tensor<T>::compute_strides()
    {
        strides_.resize(shape_.size());
        size_t stride = 1;
        for (int i = shape_.size() - 1; i >= 0; --i)
        {
            strides_[i] = stride;
            stride *= shape_[i];
        }
    }

    // Compute offset method
    template <typename T>
    size_t Tensor<T>::compute_offset(const std::vector<size_t> &indices) const
    {
        size_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i)
            offset += indices[i] * strides_[i];
        return offset;
    }

    // Overdrive operator() for easy indexing
    template <typename T>
    T &Tensor<T>::operator()(const std::vector<size_t> &indices)
    {
        return buffer_->data()[compute_offset(indices)];
    }

    template <typename T>
    const T &Tensor<T>::operator()(const std::vector<size_t> &indices) const
    {
        return buffer_->data()[compute_offset(indices)];
    }

    // Explicit instantiation
    template class Tensor<float>;
    template class Tensor<double>;

}
