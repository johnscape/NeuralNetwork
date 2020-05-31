#include "FeedForwardLayer.h"
#include "Optimizer.h"

FeedForwardLayer::FeedForwardLayer(Layer* inputLayer, unsigned int count) : Layer(inputLayer, count)
{
	Weights = new Matrix(inputLayer->GetSize(), count);
	Output = new Matrix(1, count);
	Bias = new Matrix(1, count);
	InnerState = new Matrix(1, count);
	WeightError = new Matrix(inputLayer->GetSize(), count);
	LayerError = new Matrix(1, inputLayer->GetSize());
	BiasError = new Matrix(1, count);
	function = new TanhFunction();

	MatrixMath::FillWith(Bias, 1);
}

FeedForwardLayer::~FeedForwardLayer()
{
	if (function)
		delete function;
	delete BiasError;
}

void FeedForwardLayer::SetInput(Layer* input)
{
	inputLayer = input;
	Weights = new Matrix(input->OutputSize(), Size);
}

void FeedForwardLayer::Compute()
{
	MatrixMath::FillWith(InnerState, 0);
	Matrix* prev_out = inputLayer->GetOutput();
	MatrixMath::Multiply(prev_out, Weights, InnerState);
	MatrixMath::AddIn(InnerState, Bias);
	function->CalculateInto(InnerState, Output);
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

void FeedForwardLayer::GetBackwardPass(Matrix* error, bool recursive)
{
	Matrix* derivate = function->CalculateDerivateMatrix(Output);
	MatrixMath::FillWith(LayerError, 0);

	for (unsigned int neuron = 0; neuron < Size; neuron++)
	{
		float delta = error->GetValue(neuron);
		delta *= derivate->GetValue(neuron);
		for (unsigned int incoming = 0; incoming < inputLayer->GetSize(); incoming++)
		{
			float wt = inputLayer->GetFrozenOutput()->GetValue(incoming) * delta;
			WeightError->SetValue(incoming, neuron, wt);
			LayerError->AdjustValue(incoming, delta * Weights->GetValue(incoming, neuron));
		}

		BiasError->SetValue(neuron, delta);
	}

	delete derivate;

	if (recursive)
		inputLayer->GetBackwardPass(LayerError);
}

void FeedForwardLayer::Train(Optimizer* optimizer)
{
	optimizer->ModifyWeights(Weights, WeightError);
	optimizer->ModifyWeights(Bias, BiasError);
}
