#include "Optimizer.h"

Optimizer::Optimizer(LossFuction loss, LossDerivate derivate, Layer* output)
{
	this->loss = loss;
	this->derivate = derivate;
	this->outputLayer = output;
}

Optimizer::~Optimizer()
{
	for (unsigned int i = 0; i < errors.size(); i++)
		delete errors[i];
	errors.clear();
}

void Optimizer::CalculateErrors(Layer* currentLayer)
{
	Matrix* layerOutput = currentLayer->GetOutput();
}

void Optimizer::CalculateOutputError(Matrix* output, Matrix* expected)
{
	ClearErrors();
	Matrix* outputError = new Matrix(1, output->GetColumnCount());
	for (unsigned int i = 0; i < output->GetColumnCount(); i++)
	{
		outputError->SetValue(i, derivate(output, expected, i));
	}
	errors.push_back(outputError);
}

void Optimizer::ClearErrors()
{
	for (unsigned int i = 0; i < errors.size(); i++)
		delete errors[i];
	errors.clear();
}
