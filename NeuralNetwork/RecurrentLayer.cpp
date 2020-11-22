#include "RecurrentLayer.h"

RecurrentLayer::RecurrentLayer(std::shared_ptr<Layer> inputLayer, unsigned int size, unsigned int timeSteps) :
	Layer(inputLayer), TimeSteps(timeSteps), CurrentStep(0), Size(size)
{
	Weights.reset(new Matrix(inputLayer->GetOutput()->GetVectorSize(), size));
	Output.reset(new Matrix(1, size));
	Bias.reset(new Matrix(1, size));
	InnerState.reset(new Matrix(1, size));
	WeightError.reset(new Matrix(inputLayer->GetOutput()->GetVectorSize(), size));
	LayerError.reset(new Matrix(1, inputLayer->GetOutput()->GetVectorSize()));
	SavedState.reset(new Matrix(1, size));
	BiasError.reset(new Matrix(1, size));
	RecursiveWeight.reset(new Matrix(size, size));
	RecursiveWeightError.reset(new Matrix(size, size));
	function = new TanhFunction();

	MatrixMath::FillWith(Bias, 1);
	MatrixMath::FillWith(Weights, 1);
}

RecurrentLayer::~RecurrentLayer()
{
	Weights.reset();
	Bias.reset();
	RecursiveWeight.reset();
	InnerState.reset();
	SavedState.reset();

	WeightError.reset();
	BiasError.reset();
	RecursiveWeightError.reset();


	while (!TrainingStates.empty())
		TrainingStates.pop();
}

void RecurrentLayer::Compute()
{
	MatrixMath::FillWith(InnerState, 0);
	LayerInput->Compute();
	MatrixMath::Multiply(LayerInput->GetOutput(), Weights, InnerState);
	MatrixMath::Multiply(SavedState, RecursiveWeight, InnerState);
	MatrixMath::AddIn(InnerState, Bias);
	MatrixMath::Copy(InnerState, SavedState);
	if (TrainingMode)
	{
		std::shared_ptr<Matrix> tmp(new Matrix(*InnerState));
		TrainingStates.push(tmp);
		if (TrainingStates.size() > TimeSteps)
			TrainingStates.pop();
	}
	function->CalculateInto(InnerState, Output);
}

std::shared_ptr<Matrix> RecurrentLayer::GetOutput()
{
	return Output;
}

std::shared_ptr<Matrix> RecurrentLayer::ComputeAndGetOutput()
{
	Compute();
	return Output;
}

void RecurrentLayer::GetBackwardPass(std::shared_ptr<Matrix> error, bool recursive)
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
			WeightError->AdjustValue(incoming, neuron, wt);
			LayerError->AdjustValue(incoming, delta * Weights->GetValue(incoming, neuron));
		}
		for (unsigned int hidden = 0; hidden < Size; hidden++)
		{

		}
		for (unsigned int time = 0; time < TimeSteps; time++)
		{

		}

		BiasError->SetValue(neuron, delta);
	}

	derivate.reset();

	if (recursive)
		LayerInput->GetBackwardPass(LayerError);
}

void RecurrentLayer::Train(Optimizer* optimizer)
{
}

void RecurrentLayer::SetTrainingMode(bool mode)
{
	TrainingMode = mode;
	if (!mode)
	{
		while (!TrainingStates.empty())
			TrainingStates.pop();
	}
}

