#pragma once
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include "tensor.hpp"
#include "backend.hpp"

namespace dl {

class Node {
public:
    std::string op_name;
    std::vector<std::shared_ptr<Edge>> inputs;
    std::vector<std::shared_ptr<Edge>> outputs;
    KernelAttributes attrs;

    Node(const std::string& op_name,
         std::vector<std::shared_ptr<Edge>> inputs,
         std::vector<std::shared_ptr<Edge>> outputs,
         const KernelAttributes& attrs = {})
        : op_name(op_name),
          inputs(std::move(inputs)),
          outputs(std::move(outputs)),
          attrs(attrs) {}
};

class Edge {
public:
    std::shared_ptr<Tensor> tensor;

    explicit Edge(std::shared_ptr<Tensor> tensor)
        : tensor(std::move(tensor)) {}
};

class Graph {
public:
    std::vector<std::shared_ptr<Node>> nodes;
    std::vector<std::shared_ptr<Edge>> edges;

    void add_node(const std::shared_ptr<Node>& node) {
        nodes.push_back(node);
    }

    void add_edge(const std::shared_ptr<Edge>& edge) {
        edges.push_back(edge);
    }
};

} // namespace dl
