#ifndef FPIE_CORE_CUDA_UTILS_H_
#define FPIE_CORE_CUDA_UTILS_H_

#include "cuda_to_hip.h"

// ops
inline __host__ __device__ int4 operator*(int4 a, int b) {
  return make_int4(a.x * b, a.y * b, a.z * b, a.w * b);
}

#if !defined(USE_HIP)
// HIP's vector types already define a componentwise operator+= for float3;
// providing our own would make every `+=` ambiguous. CUDA's float3 is a plain
// struct with no operators, so it still needs this. Same semantics either way.
inline __host__ __device__ void operator+=(float3& a, float3 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}
#endif

inline __host__ __device__ float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
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

void print_cuda_info();

#endif  // FPIE_CORE_CUDA_UTILS_H_
