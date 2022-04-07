cmake_minimum_required(VERSION 3.4)

project(pie_core)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(pybind11)

include_directories(pie/core)

find_package(OpenMP)
if(OpenMP_FOUND)
  add_subdirectory(pie/core/openmp)
endif()

find_library(CUDA_LIB cuda)
if(CUDA_LIB)
  add_subdirectory(pie/core/cuda)
endif()
