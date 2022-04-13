#ifndef FPIE_CORE_MPI_HELPER_H_
#define FPIE_CORE_MPI_HELPER_H_

#include <tuple>

#include "solver.h"

class MPIEquSolver : public EquSolver {
  int* maskbuf;
  unsigned char* imgbuf;
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

class MPIGridSolver : public GridSolver {
  unsigned char* imgbuf;
  int proc_id, n_proc, min_interval, *offset;

 public:
  explicit MPIGridSolver(int min_interval);

  ~MPIGridSolver();

  void post_reset();

  void sync();

  inline void update_equation(int id);

  void calc_error();

  std::tuple<py::array_t<unsigned char>, py::array_t<float>> step(
      int iteration);
};

#endif  // FPIE_CORE_MPI_HELPER_H_
