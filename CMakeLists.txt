cmake_minimum_required(VERSION 3.4)

project(pie_core)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -std=c++1y -fPIC -O3 -fomit-frame-pointer -D__STDC_CONSTANT_MACROS -lpthread -fopenmp"
)

add_subdirectory(pybind11)

set(LINK_LIBS dl rt z)

pybind11_add_module(pie_core_openmp pie_core/openmp.cc)
target_link_libraries(pie_core_openmp PRIVATE ${LINK_LIBS})
