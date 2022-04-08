#include "solver.h"

#include <tuple>

#include "helper.h"

PYBIND11_MODULE(pie_core_cuda, m) {
  py::class_<CudaSolver>(m, "Solver")
      .def(py::init<>())
      .def("partition", &CudaSolver::partition)
      .def("reset", &CudaSolver::reset)
      .def("step", &CudaSolver::step);
}
