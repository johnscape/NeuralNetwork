# CMakeList.txt : CMake project for NeuralNetwork, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.18)

project ("NeuralNetwork"	VERSION 0.0.1
							DESCRIPTION "A C++ library for machine learning purposes."
							LANGUAGES CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(GPU_SETTING "CUDA" CACHE STRING "Select type of GPU support")
set(GPU_SETTING_VALUES "None;CUDA;OpenCL")

set_property(CACHE GPU_SETTING PROPERTY STRINGS ${GPU_SETTING_VALUES})

#option	(USE_CUDA "Enable to build CUDA based computation" OFF)
option	(BUILD_TESTS "Build tests" ON)
option	(BUILD_DOC "Build documentation" ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4")

include(FetchContent)

if (${GPU_SETTING} STREQUAL "None")
    set (GPU_TYPE 0)
elseif (${GPU_SETTING} STREQUAL "CUDA")
    enable_language(CUDA)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.1)
    include_directories("/usr/local/cuda-12.1")
    find_package(CUDA REQUIRED)
    set (GPU_TYPE 1)
    find_package(CUDAToolkit)
else()
    set (GPU_TYPE 0)
    message ("OpenCL support is not implemented yet, defaulting to CPU build")
endif()

configure_file(
	"${PROJECT_SOURCE_DIR}/include/NeuralNetwork/Constants.h.in"
	"${PROJECT_SOURCE_DIR}/include/NeuralNetwork/Constants.h"
)

add_subdirectory(src)

if (BUILD_DOC)
        message("Building docs...")
	add_subdirectory(docs)
endif()

if(BUILD_TESTS)
        message("Building tests...")
	add_subdirectory(tests)
endif()