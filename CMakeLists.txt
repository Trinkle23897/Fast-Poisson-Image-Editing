cmake_minimum_required(VERSION 3.4)

project(pie_core)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -std=c++1y -fPIC -O3 -fomit-frame-pointer -D__STDC_CONSTANT_MACROS -lpthread"
)

add_subdirectory(pybind11)

include_directories(pie)

set(LINK_LIBS dl rt z)

pybind11_add_module(pie_core pie_core/pie.cc)
target_link_libraries(pie_core PRIVATE ${LINK_LIBS})
