#pragma once
#include <memory>
#include "graph.hpp"
#include "compiler.hpp"
#include "execution_plan.hpp"

namespace dl {

class Model {
public:
    Model() : executionPlan(nullptr) {}

    void compile(std::shared_ptr<Compiler> compiler) {
        executionPlan = compiler->compile(graph);
    }

    void fit() {
        if (executionPlan) {
            executionPlan->execute();
        }
    }

    Graph& get_graph() { return graph; }

private:
    Graph graph;
    std::shared_ptr<ExecutionPlan> executionPlan;
};

} // namespace dl
