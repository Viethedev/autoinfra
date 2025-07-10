#pragma once
#include <cstddef>

/**
 * @brief Represents a slice in one dimension: start:stop:step.
 *
 * Example:
 *   Range(0, 5, 2) -> 0, 2, 4
 */
struct Range
{
    size_t start_;
    size_t stop_;
    size_t step_;

    Range(size_t start, size_t stop, size_t step = 1)
        : start_(start), stop_(stop), step_(step) {}
};
