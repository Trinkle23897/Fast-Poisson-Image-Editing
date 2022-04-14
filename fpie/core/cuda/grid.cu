#include "solver.h"
#include "utils.h"

CudaGridSolver::CudaGridSolver(int grid_x, int grid_y)
    : imgbuf(NULL),
      cmask(NULL),
      cimgbuf(NULL),
      ctgt(NULL),
      cgrad(NULL),
      tmp(NULL),
      GridSolver(grid_x, grid_y) {
  print_cuda_info();
  cudaMalloc(&cerr, 3 * sizeof(float));
}

CudaGridSolver::~CudaGridSolver() {
  if (imgbuf != NULL) {
    delete[] imgbuf;
  }
  if (tmp != NULL) {
    cudaFree(cmask);
    cudaFree(ctgt);
    cudaFree(cgrad);
    cudaFree(cimgbuf);
    cudaFree(tmp);
  }
  cudaFree(cerr);
}

void CudaGridSolver::post_reset() {
  if (cmask != NULL) {
    delete[] imgbuf;
    cudaFree(cmask);
    cudaFree(ctgt);
    cudaFree(cgrad);
    cudaFree(cimgbuf);
    cudaFree(tmp);
  }
  imgbuf = new unsigned char[N * m3];
  cudaMalloc(&cmask, N * M * sizeof(int));
  cudaMalloc(&ctgt, N * m3 * sizeof(float));
  cudaMalloc(&cgrad, N * m3 * sizeof(float));
  cudaMalloc(&cimgbuf, N * m3 * sizeof(unsigned char));
  cudaMalloc(&tmp, N * m3 * sizeof(float));
  cudaMemcpy(cmask, mask, N * M * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(ctgt, tgt, N * m3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(cgrad, grad, N * m3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(tmp, 0, N * m3 * sizeof(float));
}

__global__ void iter_grid_kernel(int N, int M, int* mask, float* tgt,
                                 float* grad) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < N && y < M) {
    int id = x * M + y;
    if (mask[id]) {
      int off3 = id * 3;
      int m3 = M * 3;
      float* X = tgt + off3;
      *((float3*)(tgt + off3)) =
          ((*((float3*)(grad + off3))) + (*((float3*)(X - m3))) +
           (*((float3*)(X - 3))) + (*((float3*)(X + 3))) +
           (*((float3*)(X + m3)))) /
          4.0;
    }
  }
}

__global__ void error_grid_kernel(int N, int M, int* mask, float* tgt,
                                  float* grad, float* tmp, float* err) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < N && y < M) {
    int id = x * M + y;
    if (mask[id]) {
      int off3 = id * 3;
      int m3 = M * 3;
      float* X = tgt + off3;
      float3 t = (*((float3*)(grad + off3))) + (*((float3*)(X - m3))) +
                 (*((float3*)(X - 3))) + (*((float3*)(X + 3))) +
                 (*((float3*)(X + m3))) - (*((float3*)X)) * 4.0;
      *((float3*)(tmp + off3)) = fabs(t);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    float3 t = make_float3(0.0, 0.0, 0.0);
    x = blockIdx.x * blockDim.x;
    for (int i = 0; i < blockDim.x && x < N; ++i, ++x) {
      y = blockIdx.y * blockDim.y;
      for (int j = 0; j < blockDim.y && y < M; ++j, ++y) {
        t += *((float3*)(tmp + (x * M + y) * 3));
      }
    }
    x = blockIdx.x * blockDim.x;
    y = blockIdx.y * blockDim.y;
    *((float3*)(tmp + (x * M + y) * 3)) = t;
  }
  __syncthreads();
  if (blockIdx.x == 0 && blockIdx.y == 0) {
    float3 t = make_float3(0.0, 0.0, 0.0);
    for (int i = 0, x = 0; i < gridDim.x; ++i, x += blockDim.x) {
      for (int j = 0, y = 0; j < gridDim.y; ++j, y += blockDim.y) {
        t += *((float3*)(tmp + (x * M + y) * 3));
      }
    }
    *(float3*)(err) = t;
  }
}

__global__ void copy_img_grid_kernel(int N, int M, float* tgt,
                                     unsigned char* buf) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < N && y < M) {
    int off3 = (x * M + y) * 3;
    buf[off3 + 0] = tgt[off3 + 0] < 0     ? 0
                    : tgt[off3 + 0] > 255 ? 255
                                          : tgt[off3 + 0];
    buf[off3 + 1] = tgt[off3 + 1] < 0     ? 0
                    : tgt[off3 + 1] > 255 ? 255
                                          : tgt[off3 + 1];
    buf[off3 + 2] = tgt[off3 + 2] < 0     ? 0
                    : tgt[off3 + 2] > 255 ? 255
                                          : tgt[off3 + 2];
  }
}

std::tuple<py::array_t<unsigned char>, py::array_t<float>> CudaGridSolver::step(
    int iteration) {
  cudaMemset(cerr, 0, 3 * sizeof(float));
  dim3 block_dim(grid_x, grid_y);
  dim3 grid_dim((N + grid_x - 1) / grid_x, (M + grid_y - 1) / grid_y);
  for (int i = 0; i < iteration; ++i) {
    iter_grid_kernel<<<grid_dim, block_dim>>>(N, M, cmask, ctgt, cgrad);
    // doesn't occur any numeric issue ...
    // cudaDeviceSynchronize();
  }
  cudaDeviceSynchronize();
  copy_img_grid_kernel<<<grid_dim, block_dim>>>(N, M, ctgt, cimgbuf);
  error_grid_kernel<<<grid_dim, block_dim>>>(N, M, cmask, ctgt, cgrad, tmp,
                                             cerr);
  cudaDeviceSynchronize();

  cudaMemcpy(err, cerr, 3 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(imgbuf, cimgbuf, N * m3 * sizeof(unsigned char),
             cudaMemcpyDeviceToHost);
  return std::make_tuple(py::array({N, M, 3}, imgbuf), py::array(3, err));
}
