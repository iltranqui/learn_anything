cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(conv3d_libtorch)

# Set LibTorch path
set(CMAKE_PREFIX_PATH "/home/kerrigan/libtorch_new/libtorch")

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Include directories
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

# Add executables
add_executable(conv3d_algorithm libtorch_algorithm.cpp)

# Link libraries
target_link_libraries(conv3d_algorithm ${TORCH_LIBRARIES} ${CUDA_LIBRARIES})

# Set CUDA flags
set_target_properties(conv3d_algorithm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
