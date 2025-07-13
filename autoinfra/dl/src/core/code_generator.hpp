#pragma once
#include "graph.hpp"
#include "execution_plan.hpp"

namespace dl {

// Abstract CodeGenerator
class CodeGenerator {
public:
    virtual ~CodeGenerator() = default;
    virtual std::shared_ptr<ExecutionPlan> generateCode(Graph& graph) = 0;
};

// CPU Codegen (LLVM example)
class CpuCodeGenerator : public CodeGenerator {
public:
    std::shared_ptr<ExecutionPlan> generateCode(Graph& graph) override;
};

// CUDA Codegen (NVRTC example)
class CudaCodeGenerator : public CodeGenerator {
public:
    std::shared_ptr<ExecutionPlan> generateCode(Graph& graph) override;
};

} // namespace dl
