cmake_minimum_required(VERSION 3.22.1)
project(collector_preprocessing LANGUAGES CXX)
include(FetchContent)
FetchContent_Declare(opencl_headers
    GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
    GIT_TAG v2024.10.24
    GIT_SHALLOW TRUE)
FetchContent_GetProperties(opencl_headers)
if(NOT opencl_headers_POPULATED)
    FetchContent_Populate(opencl_headers)
endif()
add_library(collector_preprocessing SHARED preprocessing.cpp)
target_include_directories(collector_preprocessing PRIVATE ${opencl_headers_SOURCE_DIR})
target_compile_features(collector_preprocessing PRIVATE cxx_std_17)
target_compile_definitions(collector_preprocessing PRIVATE CL_TARGET_OPENCL_VERSION=120)
target_link_libraries(collector_preprocessing PRIVATE dl)
target_link_options(collector_preprocessing PRIVATE "-Wl,-z,max-page-size=16384")
