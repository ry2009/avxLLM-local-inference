cmake_minimum_required(VERSION 3.25)

project(inf_eng
        VERSION 0.1.0
        LANGUAGES CXX)

option(INFENG_ENABLE_PYTHON "Build pybind11 bindings" ON)
option(INFENG_AVX2 "Build AVX2/FMA kernels" ON)
option(INFENG_AVX512 "Build AVX-512 kernels" OFF)
option(INFENG_AMX "Build AMX kernels (requires AVX-512)" OFF)
option(INFENG_ENABLE_BENCHMARKS "Build benchmarking binaries" ON)
option(INFENG_ISPC "Enable ISPC generated kernels" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif ()

include(CTest)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules")
include(DetectCPU)

find_program(CARGO_BIN cargo)
if (NOT CARGO_BIN)
    message(FATAL_ERROR "cargo executable not found. Install Rust toolchain.")
endif ()
set(CARGO_BIN "${CARGO_BIN}" CACHE FILEPATH "Cargo executable")

set(INFENG_GENERATED_DIR "${CMAKE_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${INFENG_GENERATED_DIR}/infeng/config")
set(INFENG_CPU_HEADER "${INFENG_GENERATED_DIR}/infeng/config/cpu_features.h")
infeng_generate_cpu_header("${INFENG_CPU_HEADER}")

add_subdirectory(engine)
add_subdirectory(tools)
add_subdirectory(apis)
add_subdirectory(docs)
add_subdirectory(tests)
