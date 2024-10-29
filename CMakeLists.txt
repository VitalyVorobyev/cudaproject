cmake_minimum_required(VERSION 3.18)
project(CourseraCudaProject LANGUAGES CXX CUDA)

# Specify the CUDA standard
set(CMAKE_CUDA_STANDARD 14)

find_package(CUDAToolkit REQUIRED)

set(TARGET imgedges)

# Add the executable
add_executable(${TARGET} src/main.cu)

# Specify the target properties
set_target_properties(${TARGET} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Include directories if needed
target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/common
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link libraries if needed
#target_link_libraries(${TARGET} PRIVATE
#    CUDA::cudart
#    CUDA::cublas
#    CUDA::cufft
#    CUDA::cusolver
#    CUDA::curand
#    CUDA::nppicc
#    CUDA::nppial
#    CUDA::nppist
#    CUDA::nppidei
#    CUDA::nppig
#    CUDA::nppitc
#    CUDA::npps
#)
#