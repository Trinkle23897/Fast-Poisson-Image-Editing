set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -fopenmp")

pybind11_add_module(pie_core_openmp solver.cc)
target_link_libraries(pie_core_openmp PRIVATE)
