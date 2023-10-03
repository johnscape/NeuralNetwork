Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.0.1 # or a later release
)

FetchContent_MakeAvailable(Catch2)

set(CPP_FILES
    "MatrixTests.cpp"
    "TensorTests.cpp"
    "InputLayerTests.cpp"
    "FeedForwardLayerTests.cpp"
    "RecurrentLayerTests.cpp"
    "LSTMTests.cpp"
    "ConvLayerTests.cpp"
    "ModelTests.cpp"
)

add_executable(	NeuralNetworkTest 
				"MatrixTests.cpp"
                                "TensorTests.cpp"
				"InputLayerTests.cpp"
				"FeedForwardLayerTests.cpp"
				"RecurrentLayerTests.cpp"
				"LSTMTests.cpp"
				"ConvLayerTests.cpp"
				"ModelTests.cpp")

if (${GPU_SETTING} STREQUAL "None")
    add_executable(NeuralNetworkTest 
        ${CPP_FILES}
    )
elseif (${GPU_SETTING} STREQUAL "CUDA")
    cuda_add_executable(
        ${CPP_FILES}
    )
else()
    set (GPU_TYPE 0)
    message ("OpenCL support is not implemented yet, defaulting to CPU build")
endif()

target_link_libraries(NeuralNetworkTest PUBLIC NeuralNetwork Catch2::Catch2WithMain)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
catch_discover_tests(NeuralNetworkTest)