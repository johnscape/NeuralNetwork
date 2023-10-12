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

set(BENCHMARK_FILES
    "MatrixBenchmark.cpp"
)

if (${GPU_SETTING} STREQUAL "None")
    add_executable( NeuralNetworkTest 
        ${CPP_FILES}
    )
    add_executable( NeuralNetworkBenchmark
        ${BENCHMARK_FILES}
    )
elseif (${GPU_SETTING} STREQUAL "CUDA")
    add_executable( NeuralNetworkTest
        ${CPP_FILES}
    )
    add_executable( NeuralNetworkBenchmark
            ${BENCHMARK_FILES}
    )
else()
    set (GPU_TYPE 0)
    message ("OpenCL support is not implemented yet, defaulting to CPU build")
    add_executable(NeuralNetworkTest 
            ${CPP_FILES}
    )
endif()

include_directories(${CATCH2_INCLUDE_DIRS})
target_link_libraries(NeuralNetworkTest PUBLIC NeuralNetwork Catch2::Catch2WithMain)
target_link_libraries(NeuralNetworkBenchmark PUBLIC NeuralNetwork Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(NeuralNetworkTest)