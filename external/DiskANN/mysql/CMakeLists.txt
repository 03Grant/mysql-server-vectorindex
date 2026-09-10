cmake_minimum_required(VERSION 3.15)

# MySQL-side adapter for vendored DiskANN.
# This directory is added from the top-level MySQL CMake as:
#   add_subdirectory(external/DiskANN/mysql)
# It builds the upstream DiskANN core and re-exports a stable target
# named `diskann_vendor` for storage/innobase/vec to consume.

set(DISKANN_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/..")
set(DISKANN_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/upstream")

# Build only the core library. The apps/tests/python bindings pull in extra
# dependencies that are not needed by vecindex.
set(PYBIND OFF CACHE BOOL "" FORCE)
set(UNIT_TEST OFF CACHE BOOL "" FORCE)
set(RESTAPI OFF CACHE BOOL "" FORCE)

add_subdirectory("${DISKANN_SOURCE_DIR}" "${DISKANN_BINARY_DIR}")

if (TARGET diskann_s)
  set(_DISKANN_CORE_TARGET diskann_s)
elseif (TARGET diskann)
  set(_DISKANN_CORE_TARGET diskann)
else()
  message(FATAL_ERROR "DiskANN core target was not produced by ${DISKANN_SOURCE_DIR}")
endif()

# MySQL links vecindex into shared-library test targets, so the vendored
# DiskANN static library must be compiled as PIC.
if (TARGET diskann_s)
  set_target_properties(diskann_s PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
if (TARGET diskann)
  set_target_properties(diskann PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

add_library(diskann_vendor INTERFACE)
target_link_libraries(diskann_vendor INTERFACE ${_DISKANN_CORE_TARGET})
target_include_directories(diskann_vendor INTERFACE "${DISKANN_SOURCE_DIR}/include")

message(STATUS "DiskANN adapter exporting target: diskann_vendor -> ${_DISKANN_CORE_TARGET}")
