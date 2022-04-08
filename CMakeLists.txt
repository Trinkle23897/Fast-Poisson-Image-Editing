cmake_minimum_required(VERSION 3.4)

project(pie_core)

find_package(PythonInterp 3 REQUIRED)
set(PYTHON_VERSION_FULL ${PYTHON_VERSION_STRING})
string(REGEX REPLACE "([0-9]+\\.[0-9]+).*" "\\1" PYTHON_VERSION_MINOR
                     ${PYTHON_VERSION_STRING})
find_package(PythonLibs ${PYTHON_VERSION_MINOR} REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(pybind11)

include_directories(${PYTHON_INCLUDE_DIRS} pybind11/include pie/core)

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
