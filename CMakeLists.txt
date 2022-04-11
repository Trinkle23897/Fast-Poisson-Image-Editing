cmake_minimum_required(VERSION 3.4)

project(pie_core)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(pybind11)

include_directories(pybind11/include pie/core)

add_subdirectory(pie/core/gcc)

find_package(OpenMP)
if(OpenMP_FOUND)
  add_subdirectory(pie/core/openmp)
endif()

find_package(MPI)
if(MPI_FOUND)
  add_subdirectory(pie/core/mpi)
endif()

find_package(CUDA)
if(CUDA_FOUND)
  add_subdirectory(pie/core/cuda)
endif()
