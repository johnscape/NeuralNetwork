#include <utility>

#include "NeuralNetwork/Layers/Layer.h"
#include "NeuralNetwork/Layers/LayerException.hpp"
#include "NeuralNetwork/Layers/InputLayer.h"
#include "NeuralNetwork/Layers/FeedForwardLayer.h"
#include "NeuralNetwork/Layers/RecurrentLayer.h"
#include "NeuralNetwork/Layers/LSTM.h"
#include "NeuralNetwork/Layers/ConvLayer.h"

unsigned int Layer::LayerCount = 0;

Layer::Layer(Layer* inputLayer) : TrainingMode(false), LayerError(), Output()
{
	this->LayerInput = inputLayer;
	Id = LayerCount;
	LayerCount++;
}

Layer::Layer() : TrainingMode(false), LayerError()
{
	LayerInput = nullptr;
	Id = LayerCount;
	LayerCount++;
}

Layer* Layer::Create(Layer::LayerType type, std::vector<unsigned int> size, Layer* input)
{
	if (type == LayerType::INPUT)
		return new InputLayer(std::move(size));
	if (type == LayerType::FORWARD)
		return new FeedForwardLayer(input, size[0]);
	if (type == LayerType::RECURRENT)
		return new RecurrentLayer(input, size[0]);
	if (type == LayerType::LSTM)
		return new LSTM(input, size[0]);
	if (type == LayerType::CONV)
		return new ConvLayer(input, size[0], 1);

	return nullptr;
}

Layer::~Layer()
{
}

void Layer::SetInput(Layer* input)
{
	LayerInput = input;
}

void Layer::SetInput(const Matrix& input)
{
	SetInput((Tensor)input);
}

Tensor& Layer::GetOutput()
{
	return Output;
}

unsigned int Layer::OutputSize()
{
	return Output.GetShapeAt(1);
}

Layer* Layer::GetInputLayer()
{
	return LayerInput;
}

Tensor& Layer::GetLayerError()
{
	return LayerError;
}

void Layer::SetTrainingMode(bool mode, bool recursive)
{
	TrainingMode = mode;
	if (recursive && LayerInput)
		LayerInput->SetTrainingMode(mode);
}

Layer* Layer::CreateFromJSON(const char* data, bool isFile)
{
	//rapidjson::Document doc;
	//if (!isFile)
	//	doc.Parse(data);
	//else
	//{
	//	std::ifstream r(data);
	//	rapidjson::IStreamWrapper isw(r);
	//	doc.ParseStream(isw);
	//}

	////unsigned int layerType;
	//rapidjson::Value val;
	//val = doc["layer"]["type"];
	//Layer* ret = nullptr;
	//if (val == 0)
	//	ret = new InputLayer(1);
	//else if (val == 1)
	//	ret = new FeedForwardLayer(nullptr, 1);
	//
	//if (ret)
	//	ret->LoadFromJSON(data, isFile);
	//return ret;
	return nullptr;
}

unsigned int Layer::GetId()
{
	return Id;
}

void Layer::SetId(unsigned int id)
{
	Id = id;
}
