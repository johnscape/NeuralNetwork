FetchContent_Declare(
  catch
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v2.13.1)
FetchContent_MakeAvailable(catch)

add_executable(	NeuralNetworkTest 
				"MatrixTests.cpp"
				"TensorTests.cpp"
				"InputLayerTests.cpp"
				"FeedForwardLayerTests.cpp"
				"RecurrentLayerTests.cpp"
				"LSTMTests.cpp")

target_link_libraries(NeuralNetworkTest PRIVATE NeuralNetwork Catch2::Catch2)

list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/contrib)
include(CTest)
include(Catch)
catch_discover_tests(NeuralNetworkTest)