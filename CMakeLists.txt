cmake_minimum_required(VERSION 3.15)
project(autoinfra)

set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR}/src)

# Add your library sources here if you want:
# add_library(autoinfra_lib
#     src/deep-learning/numerical/buffer.cpp
# )

# Simple test executable
add_executable(autoinfra_test
    src/main.cpp
)
target_include_directories(autoinfra_test PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)
