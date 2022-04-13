find_package(CUDA REQUIRED)

find_package(PythonInterp 3 REQUIRED)
set(PYTHON_VERSION_FULL ${PYTHON_VERSION_STRING})
string(REGEX REPLACE "([0-9]+\\.[0-9]+).*" "\\1" PYTHON_VERSION_MINOR
                     ${PYTHON_VERSION_STRING})
find_package(PythonLibs ${PYTHON_VERSION_MINOR} REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

cuda_add_library(cudasolver SHARED equ.cu grid.cu util.cu)

pybind11_add_module(core_cuda solver.cc)
target_link_libraries(core_cuda PRIVATE cudasolver)
