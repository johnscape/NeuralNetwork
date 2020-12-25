#include "FeedForwardLayer.h"
#include "Optimizer.h"

FeedForwardLayer::FeedForwardLayer(Layer* inputLayer, unsigned int count) : Layer(inputLayer), Size(count)
{
	Weights = new Matrix(LayerInput->GetOutput()->GetVectorSize(), count);
	Output = new Matrix(1, count);
	Bias = new Matrix(1, count);
	InnerState = new Matrix(1, count);
	WeightError = new Matrix(LayerInput->GetOutput()->GetVectorSize(), count);
	LayerError = new Matrix(1, LayerInput->GetOutput()->GetVectorSize());
	BiasError = new Matrix(1, count);
	function = new TanhFunction();

	MatrixMath::FillWith(Bias, 1);
}

Layer* FeedForwardLayer::Clone()
{
	FeedForwardLayer* l = new FeedForwardLayer(LayerInput, Size);

	MatrixMath::Copy(Weights, l->GetWeights());
	MatrixMath::Copy(Output, l->GetOutput());
	MatrixMath::Copy(Bias, l->GetBias());

	l->SetActivationFunction(function);
	return l;
}

FeedForwardLayer::~FeedForwardLayer()
{
	if (function)
		delete function;
	delete Weights;
	delete Bias;
	delete InnerState;

	delete BiasError;
	delete WeightError;
}

void FeedForwardLayer::SetInput(Layer* input)
{
	LayerInput = input;
	delete Weights;
	Weights = new Matrix(LayerInput->OutputSize(), Size);
}

void FeedForwardLayer::Compute()
{
	MatrixMath::FillWith(InnerState, 0);
	LayerInput->Compute();
	Matrix* prev_out = LayerInput->GetOutput();
	MatrixMath::Multiply(prev_out, Weights, InnerState);
	MatrixMath::AddIn(InnerState, Bias);
	function->CalculateInto(InnerState, Output);
}

Matrix* FeedForwardLayer::ComputeAndGetOutput()
{
	Compute();
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
		for (unsigned int incoming = 0; incoming < LayerInput->GetOutput()->GetVectorSize(); incoming++)
		{
			float wt = LayerInput->GetOutput()->GetValue(incoming) * delta;
			WeightError->SetValue(incoming, neuron, wt);
			LayerError->AdjustValue(incoming, delta * Weights->GetValue(incoming, neuron));
		}

		BiasError->SetValue(neuron, delta);
	}

	delete derivate;

	if (recursive)
		LayerInput->GetBackwardPass(LayerError);
}

void FeedForwardLayer::Train(Optimizer* optimizer)
{
	optimizer->ModifyWeights(Weights, WeightError);
	optimizer->ModifyWeights(Bias, BiasError);

	MatrixMath::FillWith(WeightError, 0);
	MatrixMath::FillWith(BiasError, 0);
}

Matrix* FeedForwardLayer::GetBias()
{
	return Bias;
}

Matrix* FeedForwardLayer::GetWeights()
{
	return Weights;
}
