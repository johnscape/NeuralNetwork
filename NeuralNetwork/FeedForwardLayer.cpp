#include "FeedForwardLayer.h"

FeedForwardLayer::FeedForwardLayer(Layer* inputLayer, unsigned int count) : Layer(inputLayer, count), size(count)
{
	weights = new Matrix(inputLayer->GetSize(), count);
	Output = new Matrix(1, count);
	bias = new Matrix(1, count);
	inner = new Matrix(1, count);

	function = new TanhFunction();
}

FeedForwardLayer::~FeedForwardLayer()
{
	if (weights)
		delete weights;
	delete bias;
	delete inner;
	if (function)
		delete function;
}

void FeedForwardLayer::SetInput(Layer* input)
{
	inputLayer = input;
	weights = new Matrix(input->OutputSize(), size);
}

void FeedForwardLayer::Compute()
{
	MatrixMath::FillWith(inner, 0);
	Matrix* prev_out = inputLayer->GetOutput();
	MatrixMath::Multiply(prev_out, weights, inner);
	MatrixMath::AddIn(inner, bias);
	function->CalculateInto(inner, Output);
}

Matrix* FeedForwardLayer::GetOutput()
{
	this->Compute();
	return Output;
}

void FeedForwardLayer::SetActivationFunction(ActivationFunction* func)
{
	if (function)
		delete function;
	function = func;
}

Matrix* FeedForwardLayer::GetWeights()
{
	return weights;
}

Matrix* FeedForwardLayer::GetBias()
{
	return bias;
}
