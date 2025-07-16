#pragma once

#include <memory>
#include <vector>
#include "cuda_alloc.hpp"
#include "include.hpp"
#include "cuda_tensor.hpp"

namespace dl {

class CudaBackend : public Backend {
public:
    explicit CudaBackend(int device_index = 0)
        : allocator_(std::make_shared<CudaAllocator>(device_index)) {}

    std::shared_ptr<Allocator> allocator() const override {
        return allocator_;
    }

    std::shared_ptr<Tensor> allocate_tensor(
        const std::vector<size_t>& shape,
        DType dtype
    ) const override {
        return std::make_shared<CudaTensor>(shape, dtype, allocator_);
    }

private:
    std::shared_ptr<Allocator> allocator_;
};

} // namespace dl