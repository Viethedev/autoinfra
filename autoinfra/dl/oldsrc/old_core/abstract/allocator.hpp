#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include "device.hpp"

namespace dl {

/**
 * Abstract Allocator interface.
 */
class Allocator {
public:
    virtual ~Allocator() = default;

    virtual void* allocate(size_t nbytes) = 0;
    virtual void free(void* ptr) = 0;
    virtual Device device() const = 0;
};

} // namespace dl
