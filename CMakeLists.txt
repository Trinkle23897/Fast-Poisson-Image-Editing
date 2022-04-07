cmake_minimum_required(VERSION 3.4)

project(pie_core)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(pybind11)
include_directories(pie/core)
add_subdirectory(pie/core/openmp)
add_subdirectory(pie/core/cuda)
