#ifndef FPIE_CORE_CUDA_CUDA_TO_HIP_H_
#define FPIE_CORE_CUDA_CUDA_TO_HIP_H_

// Single CUDA-to-HIP shim for the GPU backend. On ROCm it aliases the CUDA
// runtime spellings this backend uses to their HIP equivalents; on NVIDIA it is
// a plain include of the CUDA runtime. This is the only file that knows HIP, so
// the kernels (equ.cu, grid.cu, utils.cu) stay in CUDA spelling and the NVIDIA
// build is unchanged. Vector types (int4/float3, make_int4/make_float3) are
// provided natively by HIP, so they need no alias.

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>

#define cudaDeviceProp           hipDeviceProp_t
#define cudaDeviceSynchronize    hipDeviceSynchronize
#define cudaError_t              hipError_t
#define cudaFree                 hipFree
#define cudaGetDeviceCount       hipGetDeviceCount
#define cudaGetDeviceProperties  hipGetDeviceProperties
#define cudaMalloc               hipMalloc
#define cudaMemcpy               hipMemcpy
#define cudaMemcpyDeviceToHost   hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice   hipMemcpyHostToDevice
#define cudaMemset               hipMemset

#else  // CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#endif

#endif  // FPIE_CORE_CUDA_CUDA_TO_HIP_H_
