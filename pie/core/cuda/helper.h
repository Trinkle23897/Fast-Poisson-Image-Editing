#ifndef PIE_CORE_CUDA_HELPER_H_
#define PIE_CORE_CUDA_HELPER_H_

#include <tuple>

#include "solver.h"

void print_cuda_info();

class CudaSolver : public Solver {
 protected:
  int* buf;
  unsigned char* buf2;
  float* tmp;

 public:
  CudaSolver() : buf(NULL), buf2(NULL), tmp(NULL), Solver() {
    print_cuda_info();
  }

  ~CudaSolver() {
    if (buf != NULL) {
      delete[] buf, buf2;
    }
    if (tmp != NULL) {
      delete[] tmp;
    }
  }

  py::array_t<int> partition(py::array_t<int> mask);
  void post_reset();
  std::tuple<py::array_t<float>, py::array_t<float>> step(int iteration);

 private:
  void step_single();
  void calc_error();
};

#endif  // PIE_CORE_CUDA_HELPER_H_
