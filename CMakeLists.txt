cmake_minimum_required(VERSION 3.18)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(cuda_sgemm_optimization LANGUAGES CXX CUDA)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default CUDA compute architectures (Pascal sm_61 through Hopper sm_90)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 61 70 75 80 86 89 90)
endif()

# Optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -Wall")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Find CUDA packages
find_package(CUDAToolkit REQUIRED)

# Source files
file(GLOB SRCS "src/*.cu" "src/kernels/*.cu")

# Executable target
add_executable(sgemm_bench ${SRCS})

# Link libraries
target_link_libraries(sgemm_bench PRIVATE CUDA::cublas CUDA::cudart)

set_target_properties(sgemm_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
