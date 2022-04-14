find_package(CUDA REQUIRED)
enable_language(CUDA)

find_package(PythonInterp 3 REQUIRED)
set(PYTHON_VERSION_FULL ${PYTHON_VERSION_STRING})
string(REGEX REPLACE "([0-9]+\\.[0-9]+).*" "\\1" PYTHON_VERSION_MINOR
                     ${PYTHON_VERSION_STRING})
find_package(PythonLibs ${PYTHON_VERSION_MINOR} REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

pybind11_add_module(core_cuda solver.cc equ.cu grid.cu utils.cu)
set_target_properties(core_cuda PROPERTIES CUDA_ARCHITECTURES OFF)
target_link_libraries(core_cuda PRIVATE cudart)
