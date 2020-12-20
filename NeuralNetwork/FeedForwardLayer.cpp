#include "FeedForwardLayer.h"
#include "Optimizer.h"

FeedForwardLayer::FeedForwardLayer(Layer& inputLayer, unsigned int count) : Layer(inputLayer), Size(count)
{
	Weights.reset(new Matrix(LayerInput->GetOutput()->GetVectorSize(), count));
	Output.reset(new Matrix(1, count));
	Bias.reset(new Matrix(1, count));
	InnerState.reset(new Matrix(1, count));
	WeightError.reset(new Matrix(LayerInput->GetOutput()->GetVectorSize(), count));
	LayerError.reset(new Matrix(1, LayerInput->GetOutput()->GetVectorSize()));
	BiasError.reset(new Matrix(1, count));
	function.reset(new TanhFunction());

	MatrixMath::FillWith(Bias, 1);
}

FeedForwardLayer::~FeedForwardLayer()
{
	if (function)
		function.reset();
	Weights.reset();
	Bias.reset();
	InnerState.reset();

	BiasError.reset();
	WeightError.reset();
}

void FeedForwardLayer::SetInput(std::shared_ptr<Layer> input)
{
	LayerInput.reset();
	LayerInput = input;
	Weights.reset(new Matrix(LayerInput->OutputSize(), Size));
}

void FeedForwardLayer::Compute()
{
	MatrixMath::FillWith(InnerState, 0);
	LayerInput->Compute();
	std::shared_ptr<Matrix> prev_out = LayerInput->GetOutput();
	MatrixMath::Multiply(prev_out, Weights, InnerState);
	MatrixMath::AddIn(InnerState, Bias);
	function->CalculateInto(InnerState, Output);
}

std::shared_ptr<Matrix> FeedForwardLayer::ComputeAndGetOutput()
{
	Compute();
	return Output;
}

void FeedForwardLayer::SetActivationFunction(std::shared_ptr<ActivationFunction> func)
{
	if (function)
		function.reset();
	function = func;
}

void FeedForwardLayer::GetBackwardPass(std::shared_ptr<Matrix> error, bool recursive)
{
	std::shared_ptr<Matrix> derivate = function->CalculateDerivateMatrix(Output);
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

	derivate.reset();

	if (recursive)
		LayerInput->GetBackwardPass(LayerError);
}

void FeedForwardLayer::Train(Optimizer* optimizer)
{
	optimizer->ModifyWeights(Weights, WeightError);
	optimizer->ModifyWeights(Bias, BiasError);
}

std::shared_ptr<Matrix> FeedForwardLayer::GetBias()
{
	return Bias;
}

std::shared_ptr<Matrix> FeedForwardLayer::GetWeights()
{
	return Weights;
}
