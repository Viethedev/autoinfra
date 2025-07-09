#pragma once
#include <cstddef>
#include <iostream>

struct Range
{
    size_t start_;
    size_t stop_;
    size_t step_;
    Range(size_t start, size_t stop, size_t step = 1)
        : start_(start), stop_(stop), step_(step) {};
};