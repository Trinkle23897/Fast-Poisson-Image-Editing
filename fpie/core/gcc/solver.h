#ifndef FPIE_CORE_GCC_SOLVER_H_
#define FPIE_CORE_GCC_SOLVER_H_

#include <tuple>

#include "base_solver.h"

class GCCEquSolver : public EquSolver {
  int* maskbuf;
  unsigned char* imgbuf;

 public:
  GCCEquSolver();
  ~GCCEquSolver();

  py::array_t<int> partition(py::array_t<int> mask);
  void post_reset();

  inline void update_equation(int i);

  void calc_error();

  std::tuple<py::array_t<unsigned char>, py::array_t<float>> step(
      int iteration);
};

class GCCGridSolver : public GridSolver {
  unsigned char* imgbuf;

 public:
  GCCGridSolver(int grid_x, int grid_y);
  ~GCCGridSolver();

  void post_reset();

  inline void update_equation(int id);

  void calc_error();

  std::tuple<py::array_t<unsigned char>, py::array_t<float>> step(
      int iteration);
};

#endif  // FPIE_CORE_GCC_SOLVER_H_
