#include "solver.h"

PYBIND11_MODULE(core_mpi, m) {
  py::class_<MPIEquSolver>(m, "EquSolver")
      .def(py::init<int>())
      .def("partition", &MPIEquSolver::partition)
      .def("reset", &MPIEquSolver::reset)
      .def("sync", &MPIEquSolver::sync)
      .def("step", &MPIEquSolver::step);
  py::class_<MPIGridSolver>(m, "GridSolver")
      .def(py::init<int>())
      .def("reset", &MPIGridSolver::reset)
      .def("sync", &MPIGridSolver::sync)
      .def("step", &MPIGridSolver::step);
}
