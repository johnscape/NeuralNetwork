file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/NeuralNetwork/*.h*")

add_library(NeuralNetwork
	Matrix.cpp
	Model.cpp
	Layer.cpp
	InputLayer.cpp
	FeedForwardLayer.cpp
	RecurrentLayer.cpp
	LSTM.cpp
	Model.cpp
	Optimizer.cpp
	GradientDescent.cpp
	GeneticAlgorithm.cpp
	Tensor.cpp
	${HEADER_LIST})

target_include_directories(NeuralNetwork PUBLIC ../include)
target_compile_features(NeuralNetwork PUBLIC cxx_std_11)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST})