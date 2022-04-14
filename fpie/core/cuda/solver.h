#ifndef FPIE_CORE_CUDA_SOLVER_H_
#define FPIE_CORE_CUDA_SOLVER_H_

#include <tuple>

#include "base_solver.h"

class CudaEquSolver : public EquSolver {
 protected:
  int* maskbuf;
  unsigned char* imgbuf;
  int block_size;
  // CUDA
  int* cA;
  unsigned char* cimgbuf;
  float *cB, *cX, *cerr, *tmp;

 public:
  explicit CudaEquSolver(int block_size);
  ~CudaEquSolver();

  py::array_t<int> partition(py::array_t<int> mask);
  void post_reset();
  std::tuple<py::array_t<unsigned char>, py::array_t<float>> step(
      int iteration);
};

class CudaGridSolver : public GridSolver {
 protected:
  unsigned char* imgbuf;
  // CUDA
  int* cmask;
  unsigned char* cimgbuf;
  float *ctgt, *cgrad, *cerr, *tmp;

 public:
  CudaGridSolver(int grid_x, int grid_y);
  ~CudaGridSolver();

  void post_reset();
  std::tuple<py::array_t<unsigned char>, py::array_t<float>> step(
      int iteration);
};

#endif  // FPIE_CORE_CUDA_SOLVER_H_
