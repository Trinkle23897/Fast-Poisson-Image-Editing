#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdio.h>

#include "helper.h"

py::array_t<int> CudaSolver::partition(py::array_t<int> mask) {
  auto arr = mask.unchecked<2>();
  int n = arr.shape(0), m = arr.shape(1);
  if (buf != NULL) {
    delete[] buf, buf2;
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
  buf2 = new unsigned char[(cnt + 1) * 3];
  return py::array({n, m}, buf);
}

void CudaSolver::post_reset() {
  if (tmp != NULL) {
    delete[] tmp;
  }
  tmp = new float[N * 3];
}

void CudaSolver::step_single() {
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

void CudaSolver::calc_error() {
  for (int i = 1; i < N; ++i) {
    int off3 = i * 3;
    int off4 = i * 4;
    int id0 = A[off4 + 0] * 3;
    int id1 = A[off4 + 1] * 3;
    int id2 = A[off4 + 2] * 3;
    int id3 = A[off4 + 3] * 3;
    tmp[off3 + 0] = std::abs(
        4 * X[off3 + 0] - (X[id0 + 0] + X[id1 + 0] + X[id2 + 0] + X[id3 + 0]) -
        B[off3 + 0]);
    tmp[off3 + 1] = std::abs(
        4 * X[off3 + 1] - (X[id0 + 1] + X[id1 + 1] + X[id2 + 1] + X[id3 + 1]) -
        B[off3 + 1]);
    tmp[off3 + 2] = std::abs(
        4 * X[off3 + 2] - (X[id0 + 2] + X[id1 + 2] + X[id2 + 2] + X[id3 + 2]) -
        B[off3 + 2]);
  }
  memset(err, 0, sizeof(err));
  for (int i = 1; i < N; ++i) {
    int off3 = i * 3;
    err[0] += tmp[off3 + 0];
    err[1] += tmp[off3 + 1];
    err[2] += tmp[off3 + 2];
  }
}

std::tuple<py::array_t<float>, py::array_t<float>> CudaSolver::step(
    int iteration) {
  for (int i = 0; i < iteration; ++i) {
    step_single();
  }
  calc_error();
  for (int i = 0; i < N * 3; ++i) {
    buf2[i] = X[i] < 0 ? 0 : X[i] > 255 ? 255 : X[i];
  }
  return std::make_tuple(py::array({N, 3}, buf2), py::array(3, err));
}

void print_cuda_info() {
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);

  printf("---------------------------------------------------------\n");
  printf("Found %d CUDA devices\n", deviceCount);

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, i);
    printf("Device %d: %s\n", i, deviceProps.name);
    printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
    printf("   Global mem: %.0f MB\n",
           static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
    printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
  }
  printf("---------------------------------------------------------\n");
}
