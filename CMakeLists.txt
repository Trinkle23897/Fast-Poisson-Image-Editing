cmake_minimum_required(VERSION 3.4)

project(fpie_core)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Get pybind11
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/pybind11)
  include(cmake_modules/GitUtils.cmake)
  git_clone(
    PROJECT_NAME
    pybind11
    GIT_URL
    https://github.com/pybind/pybind11.git
    GIT_BRANCH
    stable # To use stable version
    # GIT_TAG ${PYBIND11_VERSION}
    DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR})
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pybind11)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/pybind11/include fpie/core)

add_subdirectory(fpie/core/gcc)

find_package(OpenMP)
if(OpenMP_FOUND)
  add_subdirectory(fpie/core/openmp)
endif()

find_package(MPI)
if(MPI_FOUND)
  add_subdirectory(fpie/core/mpi)
endif()

find_package(CUDA)
if(CUDA_FOUND)
  add_subdirectory(fpie/core/cuda)
endif()
