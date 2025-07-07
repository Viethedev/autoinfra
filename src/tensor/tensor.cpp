#include "tensor/tensor.hpp"
#include <iostream>

namespace dl {
    Tensor::Tensor() = default;
    void Tensor::hello() const {
        std::cout << "Hello from Tensor!" << std::endl;
    }
}
