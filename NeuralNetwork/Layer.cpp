#include "Layer.h"
#include "MatrixMath.h"
#include "LayerException.hpp"

#include "InputLayer.h"
#include "FeedForwardLayer.h"
#include "RecurrentLayer.h"
#include "LSTM.h"

unsigned int Layer::Id = 0;

Layer::Layer(Layer* inputLayer) : TrainingMode(false), LayerError(nullptr), Output(nullptr)
{
	this->LayerInput = inputLayer;
	Id++;
}

Layer::Layer() : TrainingMode(false), LayerError(nullptr), Output(nullptr)
{
	LayerInput = nullptr;
	Id++;
}

Layer* Layer::Create(unsigned int id, unsigned int size)
{
	if (id == 0)
		return new InputLayer(size);
	if (id == 1)
		return new FeedForwardLayer(nullptr, size);
	if (id == 2)
		return new RecurrentLayer(nullptr, size);
	if (id == 3)
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
