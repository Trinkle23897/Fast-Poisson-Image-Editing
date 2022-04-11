#ifndef PIE_CORE_MPI_HELPER_H_
#define PIE_CORE_MPI_HELPER_H_

#include <tuple>

#include "solver.h"

class MPIEquSolver : public EquSolver {
  int* buf;
  unsigned char* buf2;
  float* tmp;
  int proc_id, n_proc, min_interval, *offset;

 public:
  explicit MPIEquSolver(int min_interval);

  ~MPIEquSolver();

  py::array_t<int> partition(py::array_t<int> mask);

  void post_reset();

  void sync();

  inline void update_equation(int i);

  void calc_error();

  std::tuple<py::array_t<unsigned char>, py::array_t<float>> step(
      int iteration);
};

#endif  // PIE_CORE_MPI_HELPER_H_
