find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

cuda_add_library(solver SHARED cuda.cu)

pybind11_add_module(pie_core_cuda cuda.cc)
target_link_libraries(pie_core_cuda PRIVATE solver)
