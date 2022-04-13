set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -fopenmp")

pybind11_add_module(core_openmp solver.cc equ.cc grid.cc)
target_link_libraries(core_openmp PRIVATE)
