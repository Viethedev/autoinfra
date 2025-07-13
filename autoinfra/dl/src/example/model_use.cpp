#include "model.hpp"
#include "compiler.hpp"
#include "graph_optimizer.hpp"
#include "code_generator.hpp"

int main() {
    using namespace dl;

    // 1. Build graph
    Model model;
    Graph& graph = model.get_graph();

    // User code would add nodes/ops to graph
    // For demo: graph.add_node(...)

    // 2. Build compiler pipeline
    auto genericOpt = std::make_shared<GenericGraphOptimizer>();
    auto backendOpt = std::make_shared<CpuGraphOptimizer>();
    auto codegen = std::make_shared<CpuCodeGenerator>();

    Compiler compiler(genericOpt, backendOpt, codegen);

    // 3. Compile
    model.compile(std::make_shared<Compiler>(compiler));

    // 4. Fit
    model.fit();

    return 0;
}
