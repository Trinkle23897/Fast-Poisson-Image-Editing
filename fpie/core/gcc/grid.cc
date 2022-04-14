#include "solver.h"

GCCGridSolver::GCCGridSolver(int grid_x, int grid_y)
    : imgbuf(NULL), GridSolver(grid_x, grid_y) {}

GCCGridSolver::~GCCGridSolver() {
  if (imgbuf != NULL) {
    delete[] imgbuf;
  }
}

void GCCGridSolver::post_reset() {
  if (imgbuf != NULL) {
    delete[] imgbuf;
  }
  imgbuf = new unsigned char[N * m3];
}

inline void GCCGridSolver::update_equation(int id) {
  int off3 = id * 3;
  int id0 = off3 - m3;
  int id1 = off3 - 3;
  int id2 = off3 + 3;
  int id3 = off3 + m3;
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

void GCCGridSolver::calc_error() {
  memset(err, 0, sizeof(err));
  for (int id = 0; id < N * M; ++id) {
    if (mask[id]) {
      int off3 = id * 3;
      int id0 = off3 - m3;
      int id1 = off3 - 3;
      int id2 = off3 + 3;
      int id3 = off3 + m3;
      err[0] += std::abs(grad[off3 + 0] + tgt[id0 + 0] + tgt[id1 + 0] +
                         tgt[id2 + 0] + tgt[id3 + 0] - tgt[off3 + 0] * 4.0);
      err[1] += std::abs(grad[off3 + 1] + tgt[id0 + 1] + tgt[id1 + 1] +
                         tgt[id2 + 1] + tgt[id3 + 1] - tgt[off3 + 1] * 4.0);
      err[2] += std::abs(grad[off3 + 2] + tgt[id0 + 2] + tgt[id1 + 2] +
                         tgt[id2 + 2] + tgt[id3 + 2] - tgt[off3 + 2] * 4.0);
    }
  }
}

std::tuple<py::array_t<unsigned char>, py::array_t<float>> GCCGridSolver::step(
    int iteration) {
  int block_x = (N + grid_x - 1) / grid_x;
  int block_y = (M + grid_y - 1) / grid_y;

  for (int i = 0; i < iteration; ++i) {
    for (int block_id = 0; block_id < block_x * block_y; ++block_id) {
      int id_x = block_id / block_y * grid_x,
          id_y = block_id % block_y * grid_y;
      for (int j = 0, x = id_x; j < grid_x && x < N; ++j, ++x) {
        for (int k = 0, y = id_y, id = x * M + id_y; k < grid_y && y < M;
             ++k, ++y, ++id) {
          if (mask[id]) {
            update_equation(id);
          }
        }
      }
    }
  }
  calc_error();
  for (int i = 0; i < N * M * 3; ++i) {
    imgbuf[i] = tgt[i] < 0 ? 0 : tgt[i] > 255 ? 255 : tgt[i];
  }
  return std::make_tuple(py::array({N, M, 3}, imgbuf), py::array(3, err));
}
