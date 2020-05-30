#include "Layer.h"
#include "MatrixMath.h"
#include "LayerException.hpp"

unsigned int Layer::Id = 0;

Layer::Layer(Layer* inputLayer, unsigned int size)
{
	this->Output = nullptr;
	this->inputLayer = inputLayer;
	this->Size = size;

	Weights = nullptr;
	Bias = nullptr;
	InnerState = nullptr;
	WeightError = nullptr;

	Id++;
}

Layer::~Layer()
{
	if (Output)
		delete Output;
	if (Weights)
		delete Weights;
	if (Bias)
		delete Bias;
	if (InnerState)
		delete InnerState;
	if (WeightError)
		delete WeightError;
	if (LayerError)
		delete LayerError;
}

void Layer::SetInput(Layer* input)
{
	inputLayer = input;
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

unsigned int Layer::GetSize()
{
	return Size;
}

Layer* Layer::GetInputLayer()
{
	return inputLayer;
}

Matrix* Layer::GetWeights()
{
	return Weights;
}

Matrix* Layer::GetBias()
{
	return Bias;
}

Matrix* Layer::GetInnerState()
{
	return InnerState;
}

Matrix* Layer::GetFrozenOutput()
{
	return Output;
}

Matrix* Layer::GetLayerError()
{
	return LayerError;
}
