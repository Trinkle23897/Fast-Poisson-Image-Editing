#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdio.h>

#include "helper.h"

CudaSolver::CudaSolver(int block_size)
    : buf(NULL),
      buf2(NULL),
      block_size(block_size),
      cA(NULL),
      cB(NULL),
      cX(NULL),
      tmp(NULL),
      Solver() {
  print_cuda_info();
  cudaMalloc(&cerr, 3 * sizeof(float));
}

CudaSolver::~CudaSolver() {
  if (buf != NULL) {
    delete[] buf, buf2;
  }
  if (tmp != NULL) {
    cudaFree(cA);
    cudaFree(cB);
    cudaFree(cX);
    cudaFree(tmp);
  }
  cudaFree(cerr);
}

py::array_t<int> CudaSolver::partition(py::array_t<int> mask) {
  auto arr = mask.unchecked<2>();
  int n = arr.shape(0), m = arr.shape(1);
  if (buf != NULL) {
    delete[] buf, buf2;
  }
  buf = new int[n * m];
  int cnt = 0;
  // odd
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      if ((i + j) % 2 == 1) {
        if (arr(i, j) > 0) {
          buf[i * m + j] = ++cnt;
        } else {
          buf[i * m + j] = 0;
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
          buf[i * m + j] = ++cnt;
        } else {
          buf[i * m + j] = 0;
        }
      }
    }
  }
  buf2 = new unsigned char[(cnt + 1) * 3];
  return py::array({n, m}, buf);
}

void CudaSolver::post_reset() {
  if (cA != NULL) {
    cudaFree(cA);
    cudaFree(cB);
    cudaFree(cX);
    cudaFree(cbuf);
    cudaFree(tmp);
  }
  cudaMalloc(&cA, N * 4 * sizeof(int));
  cudaMalloc(&cB, N * 3 * sizeof(float));
  cudaMalloc(&cX, N * 3 * sizeof(float));
  cudaMalloc(&cbuf, N * 3 * sizeof(unsigned char));
  cudaMalloc(&tmp, N * 3 * sizeof(float));
  cudaMemcpy(cA, A, N * 4 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(cB, B, N * 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(cX, X, N * 3 * sizeof(float), cudaMemcpyHostToDevice);
}

__global__ void iter_kernel(int N0, int N1, int* A, float* B, float* X) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (N0 <= i && i < N1) {
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
}

__global__ void error_kernel(int N0, int N1, int* A, float* B, float* X,
                             float* tmp) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (N0 <= i && i < N1) {
    int off3 = i * 3;
    int off4 = i * 4;
    int id0 = A[off4 + 0] * 3;
    int id1 = A[off4 + 1] * 3;
    int id2 = A[off4 + 2] * 3;
    int id3 = A[off4 + 3] * 3;
    int t0 = 4 * X[off3 + 0] -
             (X[id0 + 0] + X[id1 + 0] + X[id2 + 0] + X[id3 + 0]) - B[off3 + 0];
    int t1 = 4 * X[off3 + 1] -
             (X[id0 + 1] + X[id1 + 1] + X[id2 + 1] + X[id3 + 1]) - B[off3 + 1];
    int t2 = 4 * X[off3 + 2] -
             (X[id0 + 2] + X[id1 + 2] + X[id2 + 2] + X[id3 + 2]) - B[off3 + 2];
    tmp[off3 + 0] = t0 > 0 ? t0 : -t0;
    tmp[off3 + 1] = t1 > 0 ? t1 : -t1;
    tmp[off3 + 2] = t2 > 0 ? t2 : -t2;
  }
}

__global__ void copy_X_kernel(int N0, int N1, float* X, unsigned char* buf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (N0 <= i && i < N1) {
    buf[i] = X[i] < 0 ? 0 : X[i] > 255 ? 255 : X[i];
  }
}

std::tuple<py::array_t<float>, py::array_t<float>> CudaSolver::step(
    int iteration) {
  cudaMemset(cerr, 0, 3 * sizeof(float));
  int grid_size;
  for (int i = 0; i < iteration; ++i) {
    grid_size = (n_mid - 1 + block_size - 1) / block_size;
    iter_kernel<<<grid_size, block_size>>>(1, n_mid, cA, cB, cX);
    cudaDeviceSynchronize();
    grid_size = (N - n_mid + block_size - 1) / block_size;
    iter_kernel<<<grid_size, block_size>>>(n_mid, N, cA, cB, cX);
    cudaDeviceSynchronize();
  }
  grid_size = (N * 3 + block_size - 1) / block_size;
  copy_X_kernel<<<grid_size, block_size>>>(0, 3 * N, cX, cbuf);
  grid_size = (N - 1 + block_size - 1) / block_size;
  error_kernel<<<grid_size, block_size>>>(1, N, cA, cB, cX, tmp);
  cudaDeviceSynchronize();
  // error_sum_kernel<<<grid_size, block_size>>>(1, N, tmp, cerr);
  // cudaDeviceSynchronize();

  cudaMemcpy(err, cerr, 3 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(buf2, cbuf, 3 * N * sizeof(unsigned char), cudaMemcpyDeviceToHost);
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
