#include "tensor.hpp"
#include <iostream>

using namespace numerical;

int main()
{
    // Create a 3x4 tensor
    Tensor<float> t({3, 4});

    // Fill data
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            t({i, j}) = static_cast<float>(i * 10 + j);

    // Print all
    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
            std::cout << t({i, j}) << " ";
        std::cout << "\n";
    }

    // Slice: rows 1 to 3, columns 0 to 2
    auto sub = t.slice(Range(1, 3), Range(0, 2));
    std::cout << "Sliced shape: ";
    for (auto s : sub.shape())
        std::cout << s << " ";
    std::cout << "\n";

    return 0;
}
