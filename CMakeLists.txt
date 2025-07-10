cmake_minimum_required(VERSION 3.12)
project(autoinfra LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options for build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Include your header-only library
include_directories(${CMAKE_SOURCE_DIR}/src/deep-learning/numerical)

# Define the test executable
add_executable(test_tensor test/test_tensor.cpp)
