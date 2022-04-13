set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

pybind11_add_module(core_gcc solver.cc equ.cc grid.cc)
target_link_libraries(core_gcc PRIVATE)
