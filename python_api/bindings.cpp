#include <pybind11/pybind11.h>
#include "tensor/tensor.hpp"

namespace py = pybind11;

PYBIND11_MODULE(dlf_py, m) {
    py::class_<dl::Tensor>(m, "Tensor")
        .def(py::init<>())
        .def("hello", &dl::Tensor::hello);
}
