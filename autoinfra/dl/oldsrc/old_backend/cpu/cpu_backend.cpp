#include "cpu_backend.hpp"
#include "cpu_tensor.hpp"
#include <stdexcept>

namespace dl {

CpuBackend::CpuBackend()
    : allocator_(std::make_shared<CpuAllocator>()) {}

std::shared_ptr<Tensor> CpuBackend::allocate_tensor(
    const std::vector<size_t>& shape,
    DType dtype
) {
    return std::make_shared<CpuTensor>(shape, dtype, allocator_);
}

std::shared_ptr<Allocator> CpuBackend::allocator() const {
    return allocator_;
}

void CpuBackend::run_kernel(
    const std::string& op_name,
    const std::vector<std::shared_ptr<Tensor>>& inputs,
    std::vector<std::shared_ptr<Tensor>>& outputs,
    const KernelAttributes& attrs
) {
    // Placeholder dispatch
    throw std::runtime_error("CpuBackend::run_kernel not implemented yet!");
}

Device CpuBackend::device() const {
    return Device(DeviceType::CPU, 0);
}

} // namespace dl
