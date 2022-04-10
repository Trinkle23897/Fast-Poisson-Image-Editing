#ifndef PIE_CORE_CUDA_HELPER_H_
#define PIE_CORE_CUDA_HELPER_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdio.h>

#include <tuple>

#include "solver.h"

// ops

inline __host__ __device__ int4 operator*(int4 a, int b) {
  return make_int4(a.x * b, a.y * b, a.z * b, a.w * b);
}

inline __host__ __device__ void operator+=(float3& a, float3 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

inline __host__ __device__ float3 operator-(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 operator*(float3 a, float b) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ float3 operator/(float3 a, float b) {
  return make_float3(a.x / b, a.y / b, a.z / b);
}

inline __host__ __device__ float3 fabs(float3 v) {
  return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));
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

class CudaEquSolver : public EquSolver {
 protected:
  int* buf;
  unsigned char* buf2;
  int grid_size, block_size;
  // CUDA
  int* cA;
  unsigned char* cbuf;
  float *cB, *cX, *cerr, *tmp;

 public:
  explicit CudaEquSolver(int block_size);
  ~CudaEquSolver();

  py::array_t<int> partition(py::array_t<int> mask);
  void post_reset();
  std::tuple<py::array_t<unsigned char>, py::array_t<float>> step(
      int iteration);
};

#endif  // PIE_CORE_CUDA_HELPER_H_
