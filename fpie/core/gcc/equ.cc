#include "solver.h"

GCCEquSolver::GCCEquSolver() : maskbuf(NULL), imgbuf(NULL), EquSolver() {}

GCCEquSolver::~GCCEquSolver() {
  if (maskbuf != NULL) {
    delete[] maskbuf, imgbuf;
  }
}

py::array_t<int> GCCEquSolver::partition(py::array_t<int> mask) {
  auto arr = mask.unchecked<2>();
  int n = arr.shape(0), m = arr.shape(1);
  if (maskbuf != NULL) {
    delete[] maskbuf;
  }
  maskbuf = new int[n * m];
  int cnt = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      if (arr(i, j) > 0) {
        maskbuf[i * m + j] = ++cnt;
      } else {
        maskbuf[i * m + j] = 0;
      }
    }
  }
  return py::array({n, m}, maskbuf);
}

void GCCEquSolver::post_reset() {
  if (imgbuf != NULL) {
    delete[] imgbuf;
  }
  imgbuf = new unsigned char[N * 3];
}

inline void GCCEquSolver::update_equation(int i) {
  int off3 = i * 3;
  int off4 = i * 4;
  int id0 = A[off4 + 0] * 3;
  int id1 = A[off4 + 1] * 3;
  int id2 = A[off4 + 2] * 3;
  int id3 = A[off4 + 3] * 3;
  X[off3 + 0] =
      (B[off3 + 0] + X[id0 + 0] + X[id1 + 0] + X[id2 + 0] + X[id3 + 0]) / 4.0;
  X[off3 + 1] =
      (B[off3 + 1] + X[id0 + 1] + X[id1 + 1] + X[id2 + 1] + X[id3 + 1]) / 4.0;
  X[off3 + 2] =
      (B[off3 + 2] + X[id0 + 2] + X[id1 + 2] + X[id2 + 2] + X[id3 + 2]) / 4.0;
}

void GCCEquSolver::calc_error() {
  memset(err, 0, sizeof(err));
  for (int i = 1; i < N; ++i) {
    int off3 = i * 3;
    int off4 = i * 4;
    int id0 = A[off4 + 0] * 3;
    int id1 = A[off4 + 1] * 3;
    int id2 = A[off4 + 2] * 3;
    int id3 = A[off4 + 3] * 3;
    err[0] += std::abs(B[off3 + 0] + X[id0 + 0] + X[id1 + 0] + X[id2 + 0] +
                       X[id3 + 0] - X[off3 + 0] * 4.0);
    err[1] += std::abs(B[off3 + 1] + X[id0 + 1] + X[id1 + 1] + X[id2 + 1] +
                       X[id3 + 1] - X[off3 + 1] * 4.0);
    err[2] += std::abs(B[off3 + 2] + X[id0 + 2] + X[id1 + 2] + X[id2 + 2] +
                       X[id3 + 2] - X[off3 + 2] * 4.0);
  }
}

std::tuple<py::array_t<unsigned char>, py::array_t<float>> GCCEquSolver::step(
    int iteration) {
  for (int i = 0; i < iteration; ++i) {
    for (int j = 1; j < N; ++j) {
      update_equation(j);
    }
  }
  calc_error();
  for (int i = 0; i < N * 3; ++i) {
    imgbuf[i] = X[i] < 0 ? 0 : X[i] > 255 ? 255 : X[i];
  }
  return std::make_tuple(py::array({N, 3}, imgbuf), py::array(3, err));
}
