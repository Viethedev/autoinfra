#pragma once
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include "tensor.hpp"

namespace dl {

class GraphNode {
public:
    std::string op_type;
    std::vector<GraphNode*> inputs;
    std::unordered_map<std::string, double> attributes;

    // For analysis and planning
    std::vector<size_t> shape;
    DType dtype;

    GraphNode(const std::string& op) : op_type(op) {}
};

class Graph {
public:
    std::vector<std::shared_ptr<GraphNode>> nodes;
    std::vector<GraphNode*> inputs;
    std::vector<GraphNode*> outputs;

    void add_node(const std::shared_ptr<GraphNode>& node) {
        nodes.push_back(node);
    }

    void print_summary() const;
};

} // namespace dl
