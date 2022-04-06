#include <omp.h>

#include "solver.h"

class OpenMPSolver : public Solver {
  int* buf;
  float* tmp;

 public:
  explicit OpenMPSolver(int n) : buf(NULL), tmp(NULL), Solver() {
    omp_set_num_threads(n);
  }

  py::array_t<int> partition(py::array_t<int> mask) {
    auto arr = mask.unchecked<2>();
    int n = arr.shape(0), m = arr.shape(1);
    if (buf != NULL) {
      delete[] buf;
    }
    buf = new int[n * m];
    int cnt = 0;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        if (arr(i, j) > 0) {
          buf[i * m + j] = ++cnt;
        } else {
          buf[i * m + j] = 0;
        }
      }
    }
    return py::array({n, m}, buf);
  }

  void post_reset() {
    if (tmp != NULL) {
      delete[] tmp;
    }
    tmp = new float[N * 3];
  }

  void step_single() {
#pragma omp parallel for schedule(static)
    for (int i = 1; i < N; ++i) {
      int off3 = i * 3;
      int off4 = i * 4;
      tmp[off3 + 0] =
          (B[off3 + 0] + X[A[off4] * 3 + 0] + X[A[off4 + 1] * 3 + 0] +
           X[A[off4 + 2] * 3 + 0] + X[A[off4 + 3] * 3 + 0]) /
          4;
      tmp[off3 + 1] =
          (B[off3 + 1] + X[A[off4] * 3 + 1] + X[A[off4 + 1] * 3 + 1] +
           X[A[off4 + 2] * 3 + 1] + X[A[off4 + 3] * 3 + 1]) /
          4;
      tmp[off3 + 2] =
          (B[off3 + 2] + X[A[off4] * 3 + 2] + X[A[off4 + 1] * 3 + 2] +
           X[A[off4 + 2] * 3 + 2] + X[A[off4 + 3] * 3 + 2]) /
          4;
    }
    memcpy(X, tmp, sizeof(int) * N * 3);
  }

  void calc_error() {
    memset(err, 0, sizeof(err));
#pragma omp parallel for schedule(static)
    for (int i = 1; i < N; ++i) {
      int off3 = i * 3;
      int off4 = i * 4;
      tmp[off3 + 0] =
          4 * X[off3 + 0] - (X[A[off4] * 3 + 0] + X[A[off4 + 1] * 3 + 0] +
                             X[A[off4 + 2] * 3 + 0] + X[A[off4 + 3] * 3 + 0]);
      tmp[off3 + 0] = std::abs(tmp[off3 + 0] - B[off3 + 0]);
      tmp[off3 + 1] =
          4 * X[off3 + 1] - (X[A[off4] * 3 + 1] + X[A[off4 + 1] * 3 + 1] +
                             X[A[off4 + 2] * 3 + 1] + X[A[off4 + 3] * 3 + 1]);
      tmp[off3 + 1] = std::abs(tmp[off3 + 1] - B[off3 + 1]);
      tmp[off3 + 2] =
          4 * X[off3 + 2] - (X[A[off4] * 3 + 2] + X[A[off4 + 1] * 3 + 2] +
                             X[A[off4 + 2] * 3 + 2] + X[A[off4 + 3] * 3 + 2]);
      tmp[off3 + 2] = std::abs(tmp[off3 + 2] - B[off3 + 2]);
    }
    for (int i = 1; i < N; ++i) {
      int off3 = i * 3;
      err[0] += tmp[off3 + 0];
      err[1] += tmp[off3 + 1];
      err[2] += tmp[off3 + 2];
    }
  }

  std::tuple<py::array_t<float>, py::array_t<float>> step(int iteration) {
    for (int i = 0; i < iteration; ++i) {
      step_single();
    }
    calc_error();
    return std::make_tuple(py::array({N, 3}, X), py::array(3, err));
  }
};

PYBIND11_MODULE(pie_core_openmp, m) {
  py::class_<OpenMPSolver>(m, "Solver")
      .def(py::init<int>())
      .def("partition", &OpenMPSolver::partition)
      .def("reset", &OpenMPSolver::reset)
      .def("step", &OpenMPSolver::step);
}
