#pragma once
#include "graph.hpp"

namespace dl {

// Base optimizer interface
class GraphOptimizer {
public:
    virtual ~GraphOptimizer() = default;
    virtual void optimize(Graph& graph) = 0;
};

// Generic (device-independent) optimizer
class GenericGraphOptimizer : public GraphOptimizer {
public:
    void optimize(Graph& graph) override;

private:
    void constant_folding(Graph& graph);
    void eliminate_dead_code(Graph& graph);
    void infer_shapes(Graph& graph);
    void canonicalize(Graph& graph);
};

// Backend-specific interface
class BackendGraphOptimizer : public GraphOptimizer {
public:
    virtual ~BackendGraphOptimizer() = default;
};

// CPU example
class CpuGraphOptimizer : public BackendGraphOptimizer {
public:
    void optimize(Graph& graph) override;

private:
    void fuse_loops(Graph& graph);
    void plan_memory(Graph& graph);
};

// CUDA example
class CudaGraphOptimizer : public BackendGraphOptimizer {
public:
    void optimize(Graph& graph) override;

private:
    void fuse_kernels(Graph& graph);
    void optimize_memory_layout(Graph& graph);
};

} // namespace dl
