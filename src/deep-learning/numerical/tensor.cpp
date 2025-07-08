#include "tensor.hpp"

namespace numerical
{

    template <typename T>
    Tensor<T>::Tensor(const std::vector<size_t> &shape, Device device)
        : shape_(shape), device_(device)
    {
        compute_strides();
        total_size_ = 1;
        for (auto s : shape_)
            total_size_ *= s;
        buffer_ = std::make_unique<CpuBuffer<T>>(total_size_); // Only CPU for now
    }

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

    template <typename T>
    size_t Tensor<T>::compute_offset(const std::vector<size_t> &indices) const
    {
        size_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i)
            offset += indices[i] * strides_[i];
        return offset;
    }

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
