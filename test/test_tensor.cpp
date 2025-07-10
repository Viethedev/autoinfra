#include <iostream>
#include "tensor.hpp"

using namespace cpu;

void print_tensor(const Tensor<float> &t)
{
    const auto &shape = t.shape();
    std::cout << "Shape: (";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        std::cout << shape[i];
        if (i + 1 < shape.size())
            std::cout << ", ";
    }
    std::cout << ")\n";

    size_t total = 1;
    for (auto s : shape)
        total *= s;

    for (size_t i = 0; i < total; ++i)
    {
        std::cout << t.raw_buffer()[i] << " ";
    }
    std::cout << "\n";
}

int main()
{
    std::cout << "==== Tensor Ops Test ====\n";

    // Create two tensors
    Tensor<float> a({2, 3});
    Tensor<float> b({2, 3});

    // Fill them
    for (size_t i = 0; i < 6; ++i)
    {
        a.raw_buffer()[i] = static_cast<float>(i + 1);  // 1 2 3 4 5 6
        b.raw_buffer()[i] = static_cast<float>(10 + i); // 10 11 12 13 14 15
    }

    std::cout << "Tensor A:\n";
    print_tensor(a);

    std::cout << "Tensor B:\n";
    print_tensor(b);

    // Element-wise ops
    auto c = a + b;
    std::cout << "A + B:\n";
    print_tensor(c);

    auto d = a - b;
    std::cout << "A - B:\n";
    print_tensor(d);

    auto e = a * b;
    std::cout << "A * B:\n";
    print_tensor(e);

    auto f = b / a;
    std::cout << "B / A:\n";
    print_tensor(f);

    // Scalar ops
    auto g = a + 5.0f;
    std::cout << "A + 5:\n";
    print_tensor(g);

    auto h = 5.0f + a;
    std::cout << "5 + A:\n";
    print_tensor(h);

    auto i = a * 2.0f;
    std::cout << "A * 2:\n";
    print_tensor(i);

    auto j = 2.0f * a;
    std::cout << "2 * A:\n";
    print_tensor(j);

    auto k = a / 2.0f;
    std::cout << "A / 2:\n";
    print_tensor(k);

    auto l = 20.0f / a;
    std::cout << "20 / A:\n";
    print_tensor(l);

    std::cout << "==== All Tests Done ====\n";
    return 0;
}
