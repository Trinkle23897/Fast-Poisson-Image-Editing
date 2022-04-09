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
    delete[] buf;
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
  return py::array({n, m}, buf);
}

void CudaSolver::post_reset() {
  if (cA != NULL) {
    delete[] buf2;
    cudaFree(cA);
    cudaFree(cB);
    cudaFree(cX);
    cudaFree(cbuf);
    cudaFree(tmp);
  }
  buf2 = new unsigned char[N * 3];
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
  int i = blockIdx.x * blockDim.x + threadIdx.x + N0;
  if (i < N1) {
    int off3 = i * 3;
    int off4 = i * 4;
    int id0 = A[off4 + 0] * 3;
    int id1 = A[off4 + 1] * 3;
    int id2 = A[off4 + 2] * 3;
    int id3 = A[off4 + 3] * 3;
    float x0 = B[off3 + 0];
    float x1 = B[off3 + 1];
    float x2 = B[off3 + 2];
    if (id0) {
      x0 += X[id0 + 0];
      x1 += X[id0 + 1];
      x2 += X[id0 + 2];
    }
    if (id1) {
      x0 += X[id1 + 0];
      x1 += X[id1 + 1];
      x2 += X[id1 + 2];
    }
    if (id2) {
      x0 += X[id2 + 0];
      x1 += X[id2 + 1];
      x2 += X[id2 + 2];
    }
    if (id3) {
      x0 += X[id3 + 0];
      x1 += X[id3 + 1];
      x2 += X[id3 + 2];
    }
    X[off3 + 0] = x0 / 4;
    X[off3 + 1] = x1 / 4;
    X[off3 + 2] = x2 / 4;
  }
}

__global__ void iter_shared_kernel(int N0, int N1, int* A, float* B, float* X) {
  __shared__ float sX[4096 * 3];  // max shared size
  int i = blockIdx.x * blockDim.x + threadIdx.x + N0;
  if (i < N1) {
    int i0 = blockIdx.x * blockDim.x + N0;
    int i1 = (1 + blockIdx.x) * blockDim.x + N0;
    if (i1 > N1) i1 = N1;
    int off0 = i0 * 3;
    int off1 = i1 * 3;
    int off3 = i * 3;
    int off4 = i * 4;

    // load X to shared mem
    // sX[0..(i1 - i0), :] = X[i0..i1, :]
    sX[off3 - off0 + 0] = X[off3 + 0];
    sX[off3 - off0 + 1] = X[off3 + 1];
    sX[off3 - off0 + 2] = X[off3 + 2];
    __syncthreads();

    int id0 = A[off4 + 0] * 3;
    int id1 = A[off4 + 1] * 3;
    int id2 = A[off4 + 2] * 3;
    int id3 = A[off4 + 3] * 3;
    float x0 = B[off3 + 0];
    float x1 = B[off3 + 1];
    float x2 = B[off3 + 2];
    if (id0) {
      if (off0 <= id0 && id0 < off1) {
        x0 += sX[id0 - off0 + 0];
        x1 += sX[id0 - off0 + 1];
        x2 += sX[id0 - off0 + 2];
      } else {
        x0 += X[id0 + 0];
        x1 += X[id0 + 1];
        x2 += X[id0 + 2];
      }
    }
    if (id1) {
      if (off0 <= id1 && id1 < off1) {
        x0 += sX[id1 - off0 + 0];
        x1 += sX[id1 - off0 + 1];
        x2 += sX[id1 - off0 + 2];
      } else {
        x0 += X[id1 + 0];
        x1 += X[id1 + 1];
        x2 += X[id1 + 2];
      }
    }
    if (id2) {
      if (off0 <= id2 && id2 < off1) {
        x0 += sX[id2 - off0 + 0];
        x1 += sX[id2 - off0 + 1];
        x2 += sX[id2 - off0 + 2];
      } else {
        x0 += X[id2 + 0];
        x1 += X[id2 + 1];
        x2 += X[id2 + 2];
      }
    }
    if (id3) {
      if (off0 <= id3 && id3 < off1) {
        x0 += sX[id3 - off0 + 0];
        x1 += sX[id3 - off0 + 1];
        x2 += sX[id3 - off0 + 2];
      } else {
        x0 += X[id3 + 0];
        x1 += X[id3 + 1];
        x2 += X[id3 + 2];
      }
    }
    X[off3 + 0] = x0 / 4;
    X[off3 + 1] = x1 / 4;
    X[off3 + 2] = x2 / 4;
  }
}

__global__ void error_kernel(int N0, int N1, int* A, float* B, float* X,
                             float* tmp) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + N0;
  if (i < N1) {
    int off3 = i * 3;
    int off4 = i * 4;
    int id0 = A[off4 + 0] * 3;
    int id1 = A[off4 + 1] * 3;
    int id2 = A[off4 + 2] * 3;
    int id3 = A[off4 + 3] * 3;
    float t0 = 4 * X[off3 + 0] -
               (X[id0 + 0] + X[id1 + 0] + X[id2 + 0] + X[id3 + 0]) -
               B[off3 + 0];
    float t1 = 4 * X[off3 + 1] -
               (X[id0 + 1] + X[id1 + 1] + X[id2 + 1] + X[id3 + 1]) -
               B[off3 + 1];
    float t2 = 4 * X[off3 + 2] -
               (X[id0 + 2] + X[id1 + 2] + X[id2 + 2] + X[id3 + 2]) -
               B[off3 + 2];
    tmp[off3 + 0] = t0 > 0 ? t0 : -t0;
    tmp[off3 + 1] = t1 > 0 ? t1 : -t1;
    tmp[off3 + 2] = t2 > 0 ? t2 : -t2;
  }
}

__global__ void error_sum_kernel(int N, int block_size, float* tmp,
                                 float* err) {
  __shared__ float sum_err[4096 * 3];  // max shared size
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  float err0 = 0, err1 = 0, err2 = 0;
  for (int i = id; i < N; i += block_size) {
    err0 += tmp[i * 3 + 0];
    err1 += tmp[i * 3 + 1];
    err2 += tmp[i * 3 + 2];
  }
  sum_err[id * 3 + 0] = err0;
  sum_err[id * 3 + 1] = err1;
  sum_err[id * 3 + 2] = err2;
  __syncthreads();
  if (id == 0) {
    err0 = err1 = err2 = 0;
    for (int i = 0; i < block_size; ++i) {
      err0 += sum_err[i * 3 + 0];
      err1 += sum_err[i * 3 + 1];
      err2 += sum_err[i * 3 + 2];
    }
    err[0] = err0;
    err[1] = err1;
    err[2] = err2;
  }
}

__global__ void copy_X_kernel(int N0, int N1, float* X, unsigned char* buf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + N0;
  if (i < N1) {
    buf[i] = X[i] < 0 ? 0 : X[i] > 255 ? 255 : X[i];
  }
}

std::tuple<py::array_t<unsigned char>, py::array_t<float>> CudaSolver::step(
    int iteration) {
  cudaMemset(cerr, 0, 3 * sizeof(float));
  int grid_size = (N - 1 + block_size - 1) / block_size;
  for (int i = 0; i < iteration; ++i) {
    iter_kernel<<<grid_size, block_size>>>(1, N, cA, cB, cX);
    // doesn't occur any numeric issue ...
    // cudaDeviceSynchronize();
  }
  cudaDeviceSynchronize();
  grid_size = (N * 3 - 3 + block_size - 1) / block_size;
  copy_X_kernel<<<grid_size, block_size>>>(3, 3 * N, cX, cbuf);
  grid_size = (N - 1 + block_size - 1) / block_size;
  error_kernel<<<grid_size, block_size>>>(1, N, cA, cB, cX, tmp);
  cudaDeviceSynchronize();
  error_sum_kernel<<<1, block_size>>>(N, block_size, tmp, cerr);
  cudaDeviceSynchronize();

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
