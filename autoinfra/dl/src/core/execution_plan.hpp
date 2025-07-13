#pragma once

namespace dl {

class ExecutionPlan {
public:
    void execute();

    // Details depend on backend
    // For CPU: function pointers to JIT compiled code
    // For CUDA: cuModule/Kernel handles

    // Example:
    // void* cpu_func_ptr;
    // CUfunction cuda_kernel;

    // Add your actual runtime data here
};

} // namespace dl
