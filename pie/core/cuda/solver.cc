#include "solver.h"

#include <tuple>

#include "helper.h"

class CudaSolver : public Solver {
  int* buf;
  unsigned char* buf2;
  float* tmp;
  int block_size;

 public:
  explicit CudaSolver(int n_cpu, int block_size)
      : buf(NULL), buf2(NULL), tmp(NULL), block_size(block_size), Solver() {
    printCudaInfo();
  }

  py::array_t<int> partition(py::array_t<int> mask) {
    auto arr = mask.unchecked<2>();
    int n = arr.shape(0), m = arr.shape(1);
    if (buf != NULL) {
      delete[] buf, buf2;
    }
    buf = new int[n * m];
    int cnt = 0;
    for (int i = 0; i < (n + block_size - 1) / block_size; ++i) {
      for (int j = 0; j < (m + block_size - 1) / block_size; ++j) {
        for (int x = i * block_size, dx = 0; dx < block_size && x < n;
             ++dx, ++x) {
          for (int y = j * block_size, dy = 0; dy < block_size && y < m;
               ++dy, ++y) {
            if (arr(x, y) > 0) {
              buf[x * m + y] = ++cnt;
            } else {
              buf[x * m + y] = 0;
            }
          }
        }
      }
    }
    buf2 = new unsigned char[(cnt + 1) * 3];
    return py::array({n, m}, buf);
  }

  void post_reset() {
    if (tmp != NULL) {
      delete[] tmp;
    }
    tmp = new float[N * 3];
  }

  void step_single() {
    for (int i = 1; i < N; ++i) {
      int off3 = i * 3;
      int off4 = i * 4;
      int id0 = A[off4 + 0] * 3;
      int id1 = A[off4 + 1] * 3;
      int id2 = A[off4 + 2] * 3;
      int id3 = A[off4 + 3] * 3;
      tmp[off3 + 0] =
          (B[off3 + 0] + X[id0 + 0] + X[id1 + 0] + X[id2 + 0] + X[id3 + 0]) / 4;
      tmp[off3 + 1] =
          (B[off3 + 1] + X[id0 + 1] + X[id1 + 1] + X[id2 + 1] + X[id3 + 1]) / 4;
      tmp[off3 + 2] =
          (B[off3 + 2] + X[id0 + 2] + X[id1 + 2] + X[id2 + 2] + X[id3 + 2]) / 4;
    }
    for (int i = 3; i < N * 3; ++i) {
      X[i] = tmp[i];
    }
  }

  void calc_error() {
    for (int i = 1; i < N; ++i) {
      int off3 = i * 3;
      int off4 = i * 4;
      int id0 = A[off4 + 0] * 3;
      int id1 = A[off4 + 1] * 3;
      int id2 = A[off4 + 2] * 3;
      int id3 = A[off4 + 3] * 3;
      tmp[off3 + 0] = std::abs(
          4 * X[off3 + 0] -
          (X[id0 + 0] + X[id1 + 0] + X[id2 + 0] + X[id3 + 0]) - B[off3 + 0]);
      tmp[off3 + 1] = std::abs(
          4 * X[off3 + 1] -
          (X[id0 + 1] + X[id1 + 1] + X[id2 + 1] + X[id3 + 1]) - B[off3 + 1]);
      tmp[off3 + 2] = std::abs(
          4 * X[off3 + 2] -
          (X[id0 + 2] + X[id1 + 2] + X[id2 + 2] + X[id3 + 2]) - B[off3 + 2]);
    }
    memset(err, 0, sizeof(err));
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
    for (int i = 0; i < N * 3; ++i) {
      buf2[i] = X[i] < 0 ? 0 : X[i] > 255 ? 255 : X[i];
    }
    return std::make_tuple(py::array({N, 3}, buf2), py::array(3, err));
  }
};

PYBIND11_MODULE(pie_core_cuda, m) {
  py::class_<CudaSolver>(m, "Solver")
      .def(py::init<int, int>())
      .def("partition", &CudaSolver::partition)
      .def("reset", &CudaSolver::reset)
      .def("step", &CudaSolver::step);
}
