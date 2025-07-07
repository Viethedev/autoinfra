cmake_minimum_required(VERSION 3.15)
project(dlf_framework)

set(CMAKE_CXX_STANDARD 17)

option(ENABLE_PYTHON_BINDINGS "Build with PyBind11" ON)

add_library(dlf_core
    src/tensor/tensor.cpp
)
target_include_directories(dlf_core PUBLIC include)

enable_testing()
find_package(Catch2 REQUIRED)
add_executable(dlf_tests tests/tensor_test.cpp)
target_link_libraries(dlf_tests PRIVATE dlf_core Catch2::Catch2WithMain)
add_test(NAME dlf_tests COMMAND dlf_tests)

if(ENABLE_PYTHON_BINDINGS)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(dlf_py python_api/bindings.cpp)
    target_link_libraries(dlf_py PRIVATE dlf_core)
endif()
