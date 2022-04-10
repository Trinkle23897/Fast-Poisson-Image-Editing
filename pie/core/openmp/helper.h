#ifndef PIE_CORE_OPENMP_HELPER_H_
#define PIE_CORE_OPENMP_HELPER_H_

#include <omp.h>

#include <tuple>

#include "solver.h"

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

#endif  // PIE_CORE_OPENMP_HELPER_H_
