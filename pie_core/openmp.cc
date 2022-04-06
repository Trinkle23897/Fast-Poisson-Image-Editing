#include <omp.h>

#include "solver.h"

class OpenMPSolver : public Solver {
 public:
  py::array_t<int> partition(py::array_t<int> mask) { return mask; }
  std::tuple<py::array_t<float>, py::array_t<float>> step(int iteration) {
    return std::make_tuple(py::array({N, 3}, X), py::array(3, err));
  }
};

PYBIND11_MODULE(pie_core_openmp, m) {
  py::class_<OpenMPSolver>(m, "Solver")
      .def(py::init<>())
      .def("partition", &OpenMPSolver::partition)
      .def("reset", &OpenMPSolver::reset)
      .def("step", &OpenMPSolver::step);
}
