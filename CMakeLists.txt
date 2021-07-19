# CMakeList.txt : CMake project for NeuralNetwork, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.14)

project ("NeuralNetwork"	VERSION 0.0.1
							DESCRIPTION "A C++ library for machine learning purposes."
							LANGUAGES CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option	(USE_CUDA "Enable to build CUDA based computation" OFF)
option	(BUILD_TESTS "Build tests" ON)
option	(BUILD_DOC "Build documentation" ON)

include(FetchContent)

configure_file(
	"${PROJECT_SOURCE_DIR}/include/NeuralNetwork/Constants.h.in"
	"${PROJECT_SOURCE_DIR}/include/NeuralNetwork/Constants.h"
)

add_subdirectory(src)

if (BUILD_DOC)
	add_subdirectory(docs)
endif()

if(BUILD_TESTS)
	add_subdirectory(tests)
endif()