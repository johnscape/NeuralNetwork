file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/NeuralNetwork/*.h*")


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

include_directories("/usr/local/cuda/include")
set(CPP_FILES
    Matrix.cpp
    TempMatrix.cpp
    Tensor.cpp
    Model.cpp
    Optimizers/GradientDescent.cpp
    Optimizers/Optimizer.cpp
    GeneticAlgorithm.cpp
    Layers/FeedForwardLayer.cpp
    Layers/Layer.cpp
    Layers/InputLayer.cpp
    Layers/RecurrentLayer.cpp
    Layers/LSTM.cpp
    Layers/ConvLayer.cpp
    ${HEADER_LIST})
    

set(CUDA_FILES
        CUDAMath.cu
        GPUActivation.cu
        ${PROJECT_SOURCE_DIR}/include/NeuralNetwork/CUDAMath.cuh
)

if (${GPU_SETTING} STREQUAL "None")
    add_library(NeuralNetwork STATIC
        ${CPP_FILES}
    )
elseif (${GPU_SETTING} STREQUAL "CUDA")
    cuda_add_library(NeuralNetwork STATIC
        ${CPP_FILES}
        ${CUDA_FILES}
    )
else()
    set (GPU_TYPE 0)
    message ("OpenCL support is not implemented yet, defaulting to CPU build")
endif()

target_include_directories(NeuralNetwork PUBLIC ../include ${rapidjson_SOURCE_DIR}/include)
target_compile_features(NeuralNetwork PUBLIC cxx_std_11)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST})