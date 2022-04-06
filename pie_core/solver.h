#ifndef __SOLVER_H__
#define __SOLVER_H__

#include <bits/stdc++.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class Solver {
 protected:
  int N, *A;
  float *B, *X, err[3];

 public:
  Solver() : N(0), A(NULL), B(NULL), X(NULL) {}
  ~Solver() {
    if (N > 0) {
      delete[] A, B, X;
    }
  }
  void reset(int n, py::array a, py::array x, py::array b) {
    if (N > 0 && n != N) {
      delete[] A, B, X;
    }
    N = n;
    A = new int[N * 4];
    B = new float[N * 3];
    X = new float[N * 3];
    // copy from numpy
    auto a_arr = a.unchecked<int, 2>();
    auto b_arr = b.unchecked<float, 2>();
    auto x_arr = x.unchecked<float, 2>();
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
  }

  virtual py::array_t<int> partition(py::array_t<int> mask) {
    throw std::runtime_error("partition not implemented");
  }

  virtual std::tuple<py::array_t<float>, py::array_t<float>> step(
      int iteration) {
    throw std::runtime_error("step not implemented");
  }
};

#endif  // __SOLVER_H__
