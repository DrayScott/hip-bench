cmake_minimum_required(VERSION 3.16)
project(hip_runtime_bench LANGUAGES CXX HIP)

set(CMAKE_CXX_COMPILER /opt/rocm-6.4.1/bin/hipcc)
set(CMAKE_HIP_COMPILER /opt/rocm-6.4.1/bin/hipcc)
set(CMAKE_HIP_ARCHITECTURES gfx1100)   # W7900

add_compile_definitions(__HIP_PLATFORM_AMD__)
include_directories(${PROJECT_SOURCE_DIR}/src)

add_executable(hip_runtime_bench
    src/main.cpp
    kernels/vector_add.hip            # can be .hip or .hip.cpp
)

