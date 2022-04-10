#ifndef PIE_CORE_SOLVER_H_
#define PIE_CORE_SOLVER_H_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <tuple>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> dsec;

namespace py = pybind11;

class EquSolver {
 protected:
  int N, *A;
  float *B, *X, err[3];

 public:
  EquSolver() : N(0), A(NULL), B(NULL), X(NULL) {}
  ~EquSolver() {
    if (N > 0) {
      delete[] A, B, X;
    }
  }
  void reset(int n, py::array_t<int> a, py::array_t<float> x,
             py::array_t<float> b) {
    if (N > 0) {
      delete[] A, B, X;
    }
    N = n;
    A = new int[N * 4];
    B = new float[N * 3];
    X = new float[N * 3];
    // copy from numpy
    auto a_arr = a.unchecked<2>();
    auto b_arr = b.unchecked<2>();
    auto x_arr = x.unchecked<2>();
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < 4; ++j) {
        A[i * 4 + j] = a_arr(i, j);
      }
    }
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < 3; ++j) {
        B[i * 3 + j] = b_arr(i, j);
      }
    }
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < 3; ++j) {
        X[i * 3 + j] = x_arr(i, j);
      }
    }
    memset(err, 0, sizeof(err));
    post_reset();
  }

  void sync() {}

  virtual py::array_t<int> partition(py::array_t<int> mask) {
    throw std::runtime_error("partition not implemented");
  }

  virtual void post_reset() {
    throw std::runtime_error("post_reset not implemented");
  }

  virtual std::tuple<py::array_t<unsigned char>, py::array_t<float>> step(
      int iteration) {
    throw std::runtime_error("step not implemented");
  }
};

class GridSolver {
 protected:
  int N, M;
};

#endif  // PIE_CORE_SOLVER_H_
