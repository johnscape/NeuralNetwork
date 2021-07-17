# CMakeList.txt : CMake project for NeuralNetwork, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("NeuralNetwork"	VERSION 0.0.1
							DESCRIPTION "A C++ library for machine learning purposes."
							LANGUAGES CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option (USE_CUDA "Enable to build CUDA based computation" OFF)
option (BUILD_TESTS "Build tests" ON)

include(FetchContent)

configure_file(
	"${PROJECT_SOURCE_DIR}/include/NeuralNetwork/Constants.h.in"
	"${PROJECT_SOURCE_DIR}/include/NeuralNetwork/Constants.h"
)

add_subdirectory(src)

if(BUILD_TESTS)
	add_subdirectory(tests)
endif()

# Add source to this project's executable.

# TODO: Add tests and install targets if needed.
