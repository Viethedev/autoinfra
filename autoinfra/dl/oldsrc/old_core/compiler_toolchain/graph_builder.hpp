#pragma once
#include <memory>
#include "graph.hpp"
#include "backend.hpp"

namespace dl {

class GraphBuilder {
private:
    std::shared_ptr<Graph> graph;
    std::shared_ptr<Backend> backend;

public:
    GraphBuilder(std::shared_ptr<Backend> backend)
        : backend(std::move(backend)),
          graph(std::make_shared<Graph>()) {}

    std::shared_ptr<Edge> input(const std::vector<size_t>& shape, DType dtype) {
        auto tensor = backend->allocate_tensor(shape, dtype);
        auto edge = std::make_shared<Edge>(tensor);
        graph->add_edge(edge);
        return edge;
    }

    std::shared_ptr<Edge> op(const std::string& op_name,
                             const std::vector<std::shared_ptr<Edge>>& inputs,
                             const KernelAttributes& attrs = {}) {
        // Allocate output tensor(s)
        std::vector<std::shared_ptr<Edge>> outputs;
        for (const auto& in : inputs) {
            auto out_tensor = backend->allocate_tensor(in->tensor->shape(), in->tensor->dtype());
            auto out_edge = std::make_shared<Edge>(out_tensor);
            graph->add_edge(out_edge);
            outputs.push_back(out_edge);
        }

        auto node = std::make_shared<Node>(op_name, inputs, outputs, attrs);
        graph->add_node(node);

        return outputs[0];  // For simplicity assume single-output
    }

    std::shared_ptr<Graph> get() const {
        return graph;
    }
};

} // namespace dl
