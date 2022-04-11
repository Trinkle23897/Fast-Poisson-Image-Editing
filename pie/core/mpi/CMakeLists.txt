find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 ${MPI_COMPILE_FLAGS}")
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})

pybind11_add_module(pie_core_mpi solver.cc equ.cc)
target_link_libraries(pie_core_mpi PRIVATE MPI::MPI_CXX)
