#include "Layer.h"
#include "MatrixMath.h"
#include "LayerException.hpp"

unsigned int Layer::Id = 0;

Layer::Layer(Layer* inputLayer, unsigned int size)
{
	this->Output = nullptr;
	this->inputLayer = inputLayer;
	this->Size = size;
	Id++;
}

Layer::~Layer()
{
	if (Output)
		delete Output;
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
