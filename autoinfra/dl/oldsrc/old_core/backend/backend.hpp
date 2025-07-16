#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include "tensor.hpp"
#include "device.hpp"
#include "dtype.hpp"
#include "allocator.hpp"

namespace dl {

struct KernelAttributes {
    std::unordered_map<std::string, double> floats;
    std::unordered_map<std::string, int> ints;
    std::unordered_map<std::string, std::string> strings;
};

class Backend {
public:
    virtual ~Backend() = default;

    // Allocate a tensor on this backend
    virtual std::shared_ptr<Tensor> allocate_tensor(
        const std::vector<size_t>& shape,
        DType dtype
    ) = 0;

    // Access allocator
    virtual std::shared_ptr<Allocator> allocator() const = 0;

    // Run a kernel
    virtual void run_kernel(
        const std::string& op_name,
        const std::vector<std::shared_ptr<Tensor>>& inputs,
        std::vector<std::shared_ptr<Tensor>>& outputs,
        const KernelAttributes& attrs = {}
    ) = 0;

    // Query device
    virtual Device device() const = 0;

    // (Optional) Compile a graph for this backend
    virtual void compile_graph(/*Graph*/ /*future*/) {
        throw std::runtime_error("Graph compilation not supported by this backend.");
    }
};

} // namespace dl
