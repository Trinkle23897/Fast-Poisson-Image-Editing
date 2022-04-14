#include "solver.h"

PYBIND11_MODULE(core_gcc, m) {
  py::class_<GCCEquSolver>(m, "EquSolver")
      .def(py::init<>())
      .def("partition", &GCCEquSolver::partition)
      .def("reset", &GCCEquSolver::reset)
      .def("sync", &GCCEquSolver::sync)
      .def("step", &GCCEquSolver::step);
  py::class_<GCCGridSolver>(m, "GridSolver")
      .def(py::init<int, int>())
      .def("reset", &GCCGridSolver::reset)
      .def("sync", &GCCGridSolver::sync)
      .def("step", &GCCGridSolver::step);
}
