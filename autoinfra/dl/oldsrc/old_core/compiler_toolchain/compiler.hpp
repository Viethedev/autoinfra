#pragma once
#include <memory>
#include "graph.hpp"
#include "graph_optimizer.hpp"
#include "code_generator.hpp"
#include "execution_plan.hpp"

namespace dl {

class Compiler {
public:
    Compiler(std::shared_ptr<GraphOptimizer> generic,
             std::shared_ptr<GraphOptimizer> backend,
             std::shared_ptr<CodeGenerator> codegen)
        : genericOptimizer(generic), backendOptimizer(backend), codeGenerator(codegen) {}

    std::shared_ptr<ExecutionPlan> compile(Graph& graph) {
        // Phase 1: Generic optimization
        genericOptimizer->optimize(graph);

        // Phase 2: Backend optimization
        backendOptimizer->optimize(graph);

        // Phase 3: Code generation
        return codeGenerator->generateCode(graph);
    }

private:
    std::shared_ptr<GraphOptimizer> genericOptimizer;
    std::shared_ptr<GraphOptimizer> backendOptimizer;
    std::shared_ptr<CodeGenerator> codeGenerator;
};

} // namespace dl
