```cmake
cmake_minimum_required(VERSION 3.16)
project(autoinfra)

set(CMAKE_CXX_STANDARD 17)

# Include headers
include_directories(include)

# Core sources
file(GLOB_RECURSE CORE_SOURCES src/core/*.cpp src/cpu/*.cpp src/cuda/*.cu)

add_library(autoinfra SHARED ${CORE_SOURCES})

# Example executable (optional)
add_executable(example examples/mnist_train.cpp)
target_link_libraries(example autoinfra)
