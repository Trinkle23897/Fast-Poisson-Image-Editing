#include "helper.h"

PYBIND11_MODULE(pie_core_mpi, m) {
  py::class_<MPIEquSolver>(m, "EquSolver")
      .def(py::init<int>())
      .def("partition", &MPIEquSolver::partition)
      .def("reset", &MPIEquSolver::reset)
      .def("sync", &MPIEquSolver::sync)
      .def("step", &MPIEquSolver::step);
}
