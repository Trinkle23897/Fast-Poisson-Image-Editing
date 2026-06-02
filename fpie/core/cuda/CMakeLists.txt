option(USE_HIP "Build the GPU backend with HIP for AMD GPUs" OFF)

find_package(PythonInterp 3 REQUIRED)
set(PYTHON_VERSION_FULL ${PYTHON_VERSION_STRING})
string(REGEX REPLACE "([0-9]+\\.[0-9]+).*" "\\1" PYTHON_VERSION_MINOR
                     ${PYTHON_VERSION_STRING})
find_package(PythonLibs ${PYTHON_VERSION_MINOR} REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

set(CUDA_SOURCES equ.cu grid.cu utils.cu)

if(USE_HIP)
  enable_language(HIP)
  # enable_language(HIP) auto-detects the installed GPUs into
  # CMAKE_HIP_ARCHITECTURES, so by default the backend builds for the GPU in this
  # machine. Fall back to gfx90a only if nothing was detected (e.g. a build host
  # with no AMD GPU); an explicit -DCMAKE_HIP_ARCHITECTURES overrides both.
  if(NOT CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "gfx90a")
  endif()
  list(REMOVE_DUPLICATES CMAKE_HIP_ARCHITECTURES)
  set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE HIP)
  # NO_EXTRAS keeps pybind11 from injecting -flto: the HIP link step does not
  # finalize LTO, so an LTO module is left without an exported PyInit_core_cuda
  # (ImportError). LTO is a non-essential size optimization here.
  pybind11_add_module(core_cuda NO_EXTRAS solver.cc ${CUDA_SOURCES})
  set_target_properties(core_cuda PROPERTIES
                        HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
  target_compile_definitions(core_cuda PRIVATE USE_HIP)
else()
  find_package(CUDA REQUIRED)
  enable_language(CUDA)
  pybind11_add_module(core_cuda solver.cc ${CUDA_SOURCES})
  set_target_properties(core_cuda PROPERTIES CUDA_ARCHITECTURES OFF)
  target_link_libraries(core_cuda PRIVATE cudart)
endif()
