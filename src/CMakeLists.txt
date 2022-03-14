file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/NeuralNetwork/*.h*")
list(REMOVE_ITEM HEADER_LIST "${PROJECT_SOURCE_DIR}/include/NeuralNetwork/Model.h")


FetchContent_Declare(
		rapidjson
		GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
		GIT_TAG master
)
FetchContent_GetProperties(rapidjson)
if(NOT rapidjson_POPULATED)
	FetchContent_Populate(rapidjson)
endif()
FetchContent_MakeAvailable(rapidjson)
find_package(RapidJSON)

add_library(NeuralNetwork
	Matrix.cpp
	TempMatrix.cpp
	Layer.cpp
	InputLayer.cpp
	Optimizer.cpp
	GeneticAlgorithm.cpp
	Tensor.cpp
	FeedForwardLayer.cpp
	GradientDescent.cpp
	RecurrentLayer.cpp
	LSTM.cpp
	${HEADER_LIST})

target_include_directories(NeuralNetwork PUBLIC ../include ${rapidjson_SOURCE_DIR}/include)
target_compile_features(NeuralNetwork PUBLIC cxx_std_11)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST})