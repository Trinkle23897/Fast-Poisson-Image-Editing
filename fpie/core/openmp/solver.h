#ifndef FPIE_CORE_OPENMP_SOLVER_H_
#define FPIE_CORE_OPENMP_SOLVER_H_

#include <tuple>

#include "base_solver.h"

class OpenMPEquSolver : public EquSolver {
  int* maskbuf;
  unsigned char* imgbuf;
  float* tmp;
  int n_mid;

 public:
  explicit OpenMPEquSolver(int n_cpu);
  ~OpenMPEquSolver();

  py::array_t<int> partition(py::array_t<int> mask);
  void post_reset();

  inline void update_equation(int i);

  void calc_error();

  std::tuple<py::array_t<unsigned char>, py::array_t<float>> step(
      int iteration);
};

class OpenMPGridSolver : public GridSolver {
  unsigned char* imgbuf;
  float* tmp;

 public:
  OpenMPGridSolver(int grid_x, int grid_y, int n_cpu);
  ~OpenMPGridSolver();

  void post_reset();

  inline void update_equation(int id);

  void calc_error();

  std::tuple<py::array_t<unsigned char>, py::array_t<float>> step(
      int iteration);
};


class OpenMPBlockRBSolver : public GridSolver {
  protected:
    unsigned char* imgbuf;
  float* tmp;
  int tile_size;

 public:
  OpenMPBlockRBSolver(int tile_size, int n_cpu);
  ~OpenMPBlockRBSolver();

  void post_reset();
  inline void update_tile(int r0, int r1, int c0, int c1);
  void calc_error();

  std::tuple<py::array_t<unsigned char>, py::array_t<float>> step(
      int iteration);
};

class OpenMPMultiSweepsRedBlackSolver : public OpenMPBlockRBSolver {
  float* tile_residuals;
  int a_max;
  float conv_threshold;

 public:
  OpenMPMultiSweepsRedBlackSolver(int tile_size, int n_cpu, int a_max,
                                   float conv_threshold);
  ~OpenMPMultiSweepsRedBlackSolver();

  void post_reset();
  void calc_tile_residuals();
  int compute_adaptive_sweeps(float initial_residual, float current_residual);

  std::tuple<py::array_t<unsigned char>, py::array_t<float>> step(
      int iteration);
};

#endif  // FPIE_CORE_OPENMP_SOLVER_H_



