cmake_minimum_required(VERSION 3.31)
project(PTS CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_library(PTS SHARED library.cu
        library.cu)

set_target_properties(PTS PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
