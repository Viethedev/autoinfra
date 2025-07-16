#pragma once
#include "backend.hpp"
#include "cpu_alloc.hpp"

namespace dl {

class CpuBackend : public Backend {
public:
    CpuBackend();

    std::shared_ptr<Tensor> allocate_tensor(
        const std::vector<size_t>& shape,
        DType dtype
    ) override;

    std::shared_ptr<Allocator> allocator() const override;

    void run_kernel(
        const std::string& op_name,
        const std::vector<std::shared_ptr<Tensor>>& inputs,
        std::vector<std::shared_ptr<Tensor>>& outputs,
        const KernelAttributes& attrs = {}
    ) override;

    Device device() const override;

private:
    std::shared_ptr<CpuAllocator> allocator_;
};

} // namespace dl
