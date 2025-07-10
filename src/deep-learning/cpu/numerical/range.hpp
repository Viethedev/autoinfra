#pragma once
#include <cstddef>

namespace cpu
{

    class Range
    {
    public:
        size_t start;
        size_t stop;
        size_t step;

        Range(size_t start_, size_t stop_, size_t step_ = 1)
            : start(start_), stop(stop_), step(step_) {}
    };

}
