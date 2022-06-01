#include "NeuralNetwork/Layers/Layer.h"
#include "NeuralNetwork/Layers/LayerException.hpp"

#include "NeuralNetwork/Layers/InputLayer.h"
//#include "NeuralNetwork/FeedForwardLayer.h"
//#include "NeuralNetwork/RecurrentLayer.h"
//#include "NeuralNetwork/LSTM.h"

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

Layer* Layer::Create(unsigned int type, unsigned int size, Layer* input)
{
	if (type == 0)
		return new InputLayer(size);
	/*if (type == 1)
		return new FeedForwardLayer(input, size);
	if (type == 2)
		return new RecurrentLayer(input, size);
	if (type == 3)
		return new LSTM(input, size);*/
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
