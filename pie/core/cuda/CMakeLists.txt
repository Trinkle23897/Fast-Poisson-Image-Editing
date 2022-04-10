find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

cuda_add_library(helper SHARED equ.cu util.cu)

pybind11_add_module(pie_core_cuda solver.cc)
target_link_libraries(pie_core_cuda PRIVATE helper)
