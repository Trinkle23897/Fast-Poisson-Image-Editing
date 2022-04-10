#ifndef PIE_CORE_CUDA_HELPER_H_
#define PIE_CORE_CUDA_HELPER_H_

#include <tuple>

#include "solver.h"

void print_cuda_info();

class CudaEquSolver : public EquSolver {
 protected:
  int* buf;
  unsigned char* buf2;
  int grid_size, block_size;
  // CUDA
  int* cA;
  unsigned char* cbuf;
  float *cB, *cX, *cerr, *tmp;

 public:
  explicit CudaEquSolver(int block_size);
  ~CudaEquSolver();

  py::array_t<int> partition(py::array_t<int> mask);
  void post_reset();
  std::tuple<py::array_t<unsigned char>, py::array_t<float>> step(
      int iteration);
};

#endif  // PIE_CORE_CUDA_HELPER_H_
