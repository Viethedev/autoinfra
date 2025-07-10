cmake_minimum_required(VERSION 3.12)
project(autoinfra LANGUAGES CXX)

# ================================
# Project Settings
# ================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ================================
# Include Directories
# ================================
include_directories(${CMAKE_SOURCE_DIR}/src/deep-learning/cpu/numerical)

# ================================
# Find OpenMP
# ================================
find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - enabling")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP not found - building without parallelization")
endif()

# ================================
# Targets
# ================================
add_executable(test_tensor test/test_tensor.cpp)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_tensor PRIVATE OpenMP::OpenMP_CXX)
endif()
