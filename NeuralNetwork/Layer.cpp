#include "Layer.h"
#include "MatrixMath.h"
#include "LayerException.hpp"

unsigned int Layer::Id = 0;

Layer::Layer(std::shared_ptr<Layer> inputLayer) : TrainingMode(false)
{
	this->Output = nullptr;
	this->LayerInput = inputLayer;
	this->LayerError = nullptr;

	Id++;
}

Layer::~Layer()
{
	if (Output)
		Output.reset();
	if (LayerError)
		LayerError.reset();
}

void Layer::SetInput(std::shared_ptr<Layer> input)
{
	if (LayerInput)
		LayerInput.reset();
	LayerInput = input;
}

std::shared_ptr<Matrix> Layer::GetOutput()
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

std::shared_ptr<Layer> Layer::GetInputLayer()
{
	return LayerInput;
}

std::shared_ptr<Matrix> Layer::GetLayerError()
{
	return LayerError;
}

void Layer::SetTrainingMode(bool mode)
{
	TrainingMode = mode;
}
