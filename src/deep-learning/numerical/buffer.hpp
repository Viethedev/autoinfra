#pragma once
#include <vector>
#include <memory>

namespace numerical {

template<typename T>
class Buffer {
public:
    virtual T* data() = 0;
    virtual const T* data() const = 0;
    virtual ~Buffer() = default;
};

template<typename T>
class CpuBuffer : public Buffer<T> {
    std::vector<T> storage_;
public:
    CpuBuffer(size_t size) : storage_(size) {}
    T* data() override { return storage_.data(); }
    const T* data() const override { return storage_.data(); }
};

}
