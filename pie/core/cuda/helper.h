#ifndef PIE_CORE_CUDA_HELPER_H_
#define PIE_CORE_CUDA_HELPER_H_

#include <tuple>

#include "solver.h"

void print_cuda_info();

class CudaSolver : public Solver {
 protected:
  int* buf;
  unsigned char* buf2;
  // CUDA
  int* cA;
  float *cB, *cX, *cerr, *tmp;

 public:
  CudaSolver();
  ~CudaSolver();

  py::array_t<int> partition(py::array_t<int> mask);
  void post_reset();
  std::tuple<py::array_t<float>, py::array_t<float>> step(int iteration);
};

#endif  // PIE_CORE_CUDA_HELPER_H_
