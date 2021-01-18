#include "Layer.h"
#include "MatrixMath.h"
#include "LayerException.hpp"

#include "InputLayer.h"
#include "FeedForwardLayer.h"
#include "RecurrentLayer.h"
#include "LSTM.h"

unsigned int Layer::LayerCount = 0;

Layer::Layer(Layer* inputLayer) : TrainingMode(false), LayerError(nullptr), Output(nullptr)
{
	this->LayerInput = inputLayer;
	Id = LayerCount;
	LayerCount++;
}

Layer::Layer() : TrainingMode(false), LayerError(nullptr), Output(nullptr)
{
	LayerInput = nullptr;
	Id = LayerCount;
	LayerCount++;
}

Layer* Layer::Create(unsigned int type, unsigned int size)
{
	if (type == 0)
		return new InputLayer(size);
	if (type == 1)
		return new FeedForwardLayer(nullptr, size);
	if (type == 2)
		return new RecurrentLayer(nullptr, size);
	if (type == 3)
		return new LSTM(nullptr, size);
	return nullptr;
}

Layer::~Layer()
{
	if (Output)
		delete Output;
	if (LayerError)
		delete LayerError;
}

void Layer::SetInput(Layer* input)
{
	LayerInput = input;
}

Matrix* Layer::GetOutput()
{
	if (!Output)
		throw LayerInputException();
	return Output;
}

unsigned int Layer::OutputSize()
{
	if (!Output || !MatrixMath::IsVector(Output))
		return -1;
	if (Output->GetRowCount() == 1)
		return Output->GetColumnCount();
	return Output->GetRowCount();
}

Layer* Layer::GetInputLayer()
{
	return LayerInput;
}

Matrix* Layer::GetLayerError()
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
	rapidjson::Document doc;
	if (!isFile)
		doc.Parse(data);
	else
	{
		std::ifstream r(data);
		rapidjson::IStreamWrapper isw(r);
		doc.ParseStream(isw);
	}

	//unsigned int layerType;
	rapidjson::Value val;
	val = doc["layer"]["type"];
	Layer* ret = nullptr;
	if (val == 0)
		ret = new InputLayer(1);
	else if (val == 1)
		ret = new FeedForwardLayer(nullptr, 1);
	
	if (ret)
		ret->LoadFromJSON(data, isFile);
	return ret;
}

unsigned int Layer::GetId()
{
	return Id;
}

void Layer::SetId(unsigned int id)
{
	Id = id;
}
