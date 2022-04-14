#ifndef FPIE_CORE_BASE_SOLVER_H_
#define FPIE_CORE_BASE_SOLVER_H_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tuple>

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
  int N, M, grid_x, grid_y, m3;
  int* mask;
  float *tgt, *grad, err[3];

 public:
  GridSolver(int grid_x, int grid_y)
      : N(0),
        M(0),
        grid_x(grid_x),
        grid_y(grid_y),
        m3(0),
        mask(NULL),
        tgt(NULL),
        grad(NULL) {}
  ~GridSolver() {
    if (N > 0) {
      delete[] mask, tgt, grad;
    }
  }

  void reset(int n, py::array_t<int> m, py::array_t<float> t,
             py::array_t<float> g) {
    if (N > 0) {
      delete[] mask, tgt, grad;
    }
    // copy from numpy
    auto mask_arr = m.unchecked<2>();
    auto tgt_arr = t.unchecked<3>();
    auto grad_arr = g.unchecked<3>();
    N = mask_arr.shape(0);
    M = mask_arr.shape(1);
    m3 = M * 3;
    mask = new int[N * M];
    tgt = new float[N * m3];
    grad = new float[N * m3];
    int ptr = 0;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
        mask[ptr++] = mask_arr(i, j);
      }
    }
    for (int i = ptr = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
        for (int k = 0; k < 3; ++k) {
          tgt[ptr++] = tgt_arr(i, j, k);
        }
      }
    }
    for (int i = ptr = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
        for (int k = 0; k < 3; ++k) {
          grad[ptr++] = grad_arr(i, j, k);
        }
      }
    }
    memset(err, 0, sizeof(err));
    post_reset();
  }

  void sync() {}

  virtual void post_reset() {
    throw std::runtime_error("post_reset not implemented");
  }

  virtual std::tuple<py::array_t<unsigned char>, py::array_t<float>> step(
      int iteration) {
    throw std::runtime_error("step not implemented");
  }
};

#endif  // FPIE_CORE_BASE_SOLVER_H_
