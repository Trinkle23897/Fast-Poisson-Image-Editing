#include <omp.h>

#include "helper.h"

OpenMPGridSolver::OpenMPGridSolver(int n_cpu)
    : imgbuf(NULL), tmp(NULL), GridSolver() {
  omp_set_num_threads(n_cpu);
}

OpenMPGridSolver::~OpenMPGridSolver() {
  if (imgbuf != NULL) {
    delete[] imgbuf, tmp;
  }
}

void OpenMPGridSolver::post_reset() {
  if (tmp != NULL) {
    delete[] tmp, imgbuf;
  }
  tmp = new float[N * M * 3];
  imgbuf = new unsigned char[N * M * 3];
  memset(tmp, 0, sizeof(float) * N * M * 3);
}

inline void OpenMPGridSolver::update_equation(int id, int x, int y) {
  int id0 = ((x - 1) * M + y) * 3;
  int id1 = ((x + 1) * M + y) * 3;
  int id2 = (x * M + y - 1) * 3;
  int id3 = (x * M + y + 1) * 3;
  int off3 = id * 3;
  tgt[off3 + 0] = (grad[off3 + 0] + tgt[id0 + 0] + tgt[id1 + 0] + tgt[id2 + 0] +
                   tgt[id3 + 0]) /
                  4.0;
  tgt[off3 + 1] = (grad[off3 + 1] + tgt[id0 + 1] + tgt[id1 + 1] + tgt[id2 + 1] +
                   tgt[id3 + 1]) /
                  4.0;
  tgt[off3 + 2] = (grad[off3 + 2] + tgt[id0 + 2] + tgt[id1 + 2] + tgt[id2 + 2] +
                   tgt[id3 + 2]) /
                  4.0;
}

void OpenMPGridSolver::calc_error() {
#pragma omp parallel for schedule(static)
  for (int id = 0; id < N * M; ++id) {
    if (mask[id]) {
      int x = id / M, y = id % M;
      int id0 = ((x - 1) * M + y) * 3;
      int id1 = ((x + 1) * M + y) * 3;
      int id2 = (x * M + y - 1) * 3;
      int id3 = (x * M + y + 1) * 3;
      int off3 = id * 3;
      tmp[off3 + 0] =
          std::abs(grad[off3 + 0] + tgt[id0 + 0] + tgt[id1 + 0] + tgt[id2 + 0] +
                   tgt[id3 + 0] - tgt[off3 + 0] * 4.0);
      tmp[off3 + 1] =
          std::abs(grad[off3 + 1] + tgt[id0 + 1] + tgt[id1 + 1] + tgt[id2 + 1] +
                   tgt[id3 + 1] - tgt[off3 + 1] * 4.0);
      tmp[off3 + 2] =
          std::abs(grad[off3 + 2] + tgt[id0 + 2] + tgt[id1 + 2] + tgt[id2 + 2] +
                   tgt[id3 + 2] - tgt[off3 + 2] * 4.0);
    }
  }
  memset(err, 0, sizeof(err));
  for (int id = 0; id < N * M; ++id) {
    int off3 = id * 3;
    err[0] += tmp[off3 + 0];
    err[1] += tmp[off3 + 1];
    err[2] += tmp[off3 + 2];
  }
}

std::tuple<py::array_t<unsigned char>, py::array_t<float>>
OpenMPGridSolver::step(int iteration) {
  for (int i = 0; i < iteration; ++i) {
#pragma omp parallel for schedule(static)
    for (int id = 0; id < N * M; ++id) {
      int x = id / M, y = id % M;
      if ((x + y) % 2 == 0 && mask[id]) {
        update_equation(id, x, y);
      }
    }
#pragma omp parallel for schedule(static)
    for (int id = 0; id < N * M; ++id) {
      int x = id / M, y = id % M;
      if ((x + y) % 2 == 1 && mask[id]) {
        update_equation(id, x, y);
      }
    }
  }
  calc_error();
#pragma omp parallel for schedule(static)
  for (int i = 0; i < N * M * 3; ++i) {
    imgbuf[i] = tgt[i] < 0 ? 0 : tgt[i] > 255 ? 255 : tgt[i];
  }
  return std::make_tuple(py::array({N, M, 3}, imgbuf), py::array(3, err));
}
