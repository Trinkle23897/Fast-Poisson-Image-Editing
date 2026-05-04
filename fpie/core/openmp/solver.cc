#include "solver.h"

PYBIND11_MODULE(core_openmp, m) {
    py::class_<OpenMPEquSolver>(m, "EquSolver")
        .def(py::init<int>())
        .def("partition", &OpenMPEquSolver::partition)
        .def("reset", &OpenMPEquSolver::reset)
        .def("sync", &OpenMPEquSolver::sync)
        .def("step", &OpenMPEquSolver::step);
    py::class_<OpenMPGridSolver>(m, "GridSolver")
        .def(py::init<int, int, int>())
        .def("reset", &OpenMPGridSolver::reset)
        .def("sync", &OpenMPGridSolver::sync)
        .def("step", &OpenMPGridSolver::step);
    py::class_<OpenMPBlockRBSolver>(m, "BlockRBSolver")
        .def(py::init<int, int>())   // tile_size, n_cpu
        .def("reset", &OpenMPBlockRBSolver::reset)
        .def("sync",  &OpenMPBlockRBSolver::sync)
        .def("step",  &OpenMPBlockRBSolver::step);
    py::class_<OpenMPMultiSweepsRedBlackSolver>(m, "MultiSweepsRedBlackSolver")
        .def(py::init<int, int, int, float>())  // tile_size, n_cpu, a_max, conv_threshold
        .def("reset", &OpenMPMultiSweepsRedBlackSolver::reset)
        .def("sync",  &OpenMPMultiSweepsRedBlackSolver::sync)
        .def("step",  &OpenMPMultiSweepsRedBlackSolver::step);
}
