cmake_minimum_required(VERSION 3.16)
project(hip_runtime_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
find_package(HIP REQUIRED)

add_executable(hip_runtime_bench src/main.cpp kernels/vector_add.hip.cpp)
target_link_libraries(hip_runtime_bench PRIVATE HIP::hip)
