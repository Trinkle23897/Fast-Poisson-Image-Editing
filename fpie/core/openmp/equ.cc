#include <omp.h>

#include "solver.h"

OpenMPEquSolver::OpenMPEquSolver(int n_cpu)
    : maskbuf(NULL), imgbuf(NULL), tmp(NULL), n_mid(0), EquSolver() {
  omp_set_num_threads(n_cpu);
}

OpenMPEquSolver::~OpenMPEquSolver() {
  if (maskbuf != NULL) {
    delete[] maskbuf, imgbuf;
  }
  if (tmp != NULL) {
    delete[] tmp;
  }
}

py::array_t<int> OpenMPEquSolver::partition(py::array_t<int> mask) {
  auto arr = mask.unchecked<2>();
  int n = arr.shape(0), m = arr.shape(1);
  if (maskbuf != NULL) {
    delete[] maskbuf;
  }
  maskbuf = new int[n * m];
  int cnt = 0;
  // odd
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      if ((i + j) % 2 == 1) {
        if (arr(i, j) > 0) {
          maskbuf[i * m + j] = ++cnt;
        } else {
          maskbuf[i * m + j] = 0;
        }
      }
    }
  }
  n_mid = cnt + 1;
  // even
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      if ((i + j) % 2 == 0) {
        if (arr(i, j) > 0) {
          maskbuf[i * m + j] = ++cnt;
        } else {
          maskbuf[i * m + j] = 0;
        }
      }
    }
  }
  return py::array({n, m}, maskbuf);
}

void OpenMPEquSolver::post_reset() {
  if (tmp != NULL) {
    delete[] tmp, imgbuf;
  }
  tmp = new float[N * 3];
  imgbuf = new unsigned char[N * 3];
}

inline void OpenMPEquSolver::update_equation(int i) {
  int off3 = i * 3;
  int off4 = i * 4;
  int id0 = A[off4 + 0] * 3;
  int id1 = A[off4 + 1] * 3;
  int id2 = A[off4 + 2] * 3;
  int id3 = A[off4 + 3] * 3;
  X[off3 + 0] =
      (B[off3 + 0] + X[id0 + 0] + X[id1 + 0] + X[id2 + 0] + X[id3 + 0]) / 4;
  X[off3 + 1] =
      (B[off3 + 1] + X[id0 + 1] + X[id1 + 1] + X[id2 + 1] + X[id3 + 1]) / 4;
  X[off3 + 2] =
      (B[off3 + 2] + X[id0 + 2] + X[id1 + 2] + X[id2 + 2] + X[id3 + 2]) / 4;
}

void OpenMPEquSolver::calc_error() {
#pragma omp parallel for schedule(static)
  for (int i = 1; i < N; ++i) {
    int off3 = i * 3;
    int off4 = i * 4;
    int id0 = A[off4 + 0] * 3;
    int id1 = A[off4 + 1] * 3;
    int id2 = A[off4 + 2] * 3;
    int id3 = A[off4 + 3] * 3;
    tmp[off3 + 0] = std::abs(B[off3 + 0] + X[id0 + 0] + X[id1 + 0] +
                             X[id2 + 0] + X[id3 + 0] - X[off3 + 0] * 4.0);
    tmp[off3 + 1] = std::abs(B[off3 + 1] + X[id0 + 1] + X[id1 + 1] +
                             X[id2 + 1] + X[id3 + 1] - X[off3 + 1] * 4.0);
    tmp[off3 + 2] = std::abs(B[off3 + 2] + X[id0 + 2] + X[id1 + 2] +
                             X[id2 + 2] + X[id3 + 2] - X[off3 + 2] * 4.0);
  }
  memset(err, 0, sizeof(err));
  for (int i = 1; i < N; ++i) {
    int off3 = i * 3;
    err[0] += tmp[off3 + 0];
    err[1] += tmp[off3 + 1];
    err[2] += tmp[off3 + 2];
  }
}

std::tuple<py::array_t<unsigned char>, py::array_t<float>>
OpenMPEquSolver::step(int iteration) {
  for (int i = 0; i < iteration; ++i) {
#pragma omp parallel for schedule(static)
    for (int j = 1; j < n_mid; ++j) {
      update_equation(j);
    }
#pragma omp parallel for schedule(static)
    for (int j = n_mid; j < N; ++j) {
      update_equation(j);
    }
  }
  calc_error();
#pragma omp parallel for schedule(static)
  for (int i = 0; i < N * 3; ++i) {
    imgbuf[i] = X[i] < 0 ? 0 : X[i] > 255 ? 255 : X[i];
  }
  return std::make_tuple(py::array({N, 3}, imgbuf), py::array(3, err));
}
