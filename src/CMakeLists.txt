file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/NeuralNetwork/*.h*")
list(REMOVE_ITEM HEADER_LIST "${PROJECT_SOURCE_DIR}/include/NeuralNetwork/Model.h")

add_library(NeuralNetwork
	Matrix.cpp
	Layer.cpp
	InputLayer.cpp
	Optimizer.cpp
	GeneticAlgorithm.cpp
	Tensor.cpp
	FeedForwardLayer.cpp
	GradientDescent.cpp
	RecurrentLayer.cpp
	LSTM.cpp
    TempMatrix.cpp
	${HEADER_LIST})

target_include_directories(NeuralNetwork PUBLIC ../include)
target_compile_features(NeuralNetwork PUBLIC cxx_std_11)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST})