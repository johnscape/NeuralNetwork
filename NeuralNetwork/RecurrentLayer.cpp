#include "RecurrentLayer.h"

RecurrentLayer::RecurrentLayer(Layer* inputLayer, unsigned int size, unsigned int timeSteps) :
	Layer(inputLayer), TimeSteps(timeSteps), CurrentStep(0), Size(size)
{
	Weights = new Matrix(LayerInput->GetOutput()->GetVectorSize(), size);
	Output = new Matrix(1, size);
	Bias = new Matrix(1, size);
	InnerState = new Matrix(1, size);
	WeightError = new Matrix(LayerInput->GetOutput()->GetVectorSize(), size);
	LayerError = new Matrix(1, LayerInput->GetOutput()->GetVectorSize());
	BiasError = new Matrix(1, size);
	RecursiveWeight = new Matrix(size, size);
	RecursiveWeightError = new Matrix(size, size);
	function = new TanhFunction();

	MatrixMath::FillWith(Bias, 1);
	MatrixMath::FillWith(Weights, 1);
}

RecurrentLayer::~RecurrentLayer()
{
	delete Weights;
	delete Bias;
	delete RecursiveWeight;
	delete InnerState;

	delete WeightError;
	delete BiasError;
	delete RecursiveWeightError;


	while (!TrainingStates.empty())
	{
		delete TrainingStates[0];
		TrainingStates.pop_front();
	}
	while (!IncomingValues.empty())
	{
		delete IncomingValues[0];
		IncomingValues.pop_front();
	}
}

void RecurrentLayer::Compute()
{
	MatrixMath::FillWith(InnerState, 0);
	LayerInput->Compute();
	MatrixMath::Multiply(LayerInput->GetOutput(), Weights, InnerState);
	MatrixMath::Multiply(Output, RecursiveWeight, InnerState);
	MatrixMath::AddIn(InnerState, Bias);
	function->CalculateInto(InnerState, Output);
	if (TrainingMode)
	{
		TrainingStates.push_back(new Matrix(*InnerState));
		IncomingValues.push_back(new Matrix(*(LayerInput->GetOutput())));
		if (TrainingStates.size() > TimeSteps)
		{
			delete TrainingStates[0];
			TrainingStates.pop_front();
			delete IncomingValues[0];
			IncomingValues.pop_front();
		}
	}
}

Matrix* RecurrentLayer::GetOutput()
{
	return Output;
}

Matrix* RecurrentLayer::ComputeAndGetOutput()
{
	Compute();
	return Output;
}

void RecurrentLayer::SetActivationFunction(ActivationFunction* func)
{
	if (function)
		delete function;
	function = func;
}

void RecurrentLayer::GetBackwardPass(Matrix* error, bool recursive)
{
	Matrix* derivate = function->CalculateDerivateMatrix(Output);
	MatrixMath::FillWith(LayerError, 0);

	std::vector<Matrix*> powers;
	for (unsigned int i = 0; i <= TimeSteps; i++)
	{
		if (!i)
			continue;
		powers.push_back(MatrixMath::Power(RecursiveWeight, i));
	}

	/*for (unsigned int neuron = 0; neuron < Size; neuron++)
	{
		float delta = error->GetValue(neuron);
		delta *= derivate->GetValue(neuron);
		for (unsigned int incoming = 0; incoming < LayerInput->GetOutput()->GetVectorSize(); incoming++)
		{
			float wt = LayerInput->GetOutput()->GetValue(incoming) * delta;
			WeightError->AdjustValue(incoming, neuron, wt);
			LayerError->AdjustValue(incoming, delta * Weights->GetValue(incoming, neuron));
		}
		for (signed int time = 0; time < TimeSteps; time++)
		{
			if (time >= TrainingStates.size()) //ennek nem k�ne megt�rt�nnie
				continue;
			for (unsigned int recursive = 0; recursive < Size; recursive++)
			{
				float rt = TrainingStates[time]->GetValue(recursive) * delta;
				if (time)
					rt *= powers[time - 1]->GetValue(recursive, neuron);
				RecursiveWeightError->AdjustValue(recursive, neuron, rt);
			}
		}

		BiasError->SetValue(neuron, delta);
	}*/
	for (unsigned int neuron = 0; neuron < Size; neuron++)
	{
		float delta = error->GetValue(neuron);
		delta *= derivate->GetValue(neuron);
		for (unsigned int time = 0; time < TimeSteps; time++)
		{
			if (TimeSteps - time >= TrainingStates.size())
				continue;
			for (unsigned int incoming = 0; incoming < LayerInput->GetOutput()->GetVectorSize(); incoming++)
			{
				float wt = 0;
				if (time)
				{
					for (unsigned int recursive = 0; recursive < Size; recursive++)
						wt += error->GetValue(recursive) * derivate->GetValue(recursive) * powers[time]->GetValue(neuron, recursive) * IncomingValues[TimeSteps - time]->GetValue(incoming);
				}
				else
					wt = IncomingValues[IncomingValues.size() - 1]->GetValue(incoming) * delta;
				WeightError->AdjustValue(incoming, neuron, wt);
				LayerError->AdjustValue(incoming, delta * Weights->GetValue(incoming, neuron));
			}
			for (unsigned int recursive = 0; recursive < Size; recursive++)
			{
				float wt = 0;
				if (time)
				{
					for (unsigned int r = 0; r < Size; r++)
						wt += error->GetValue(r) * derivate->GetValue(r) * powers[time]->GetValue(neuron, r) * TrainingStates[TimeSteps - time]->GetValue(recursive);
				}
				else
					wt = TrainingStates[TrainingStates.size() - 1]->GetValue(recursive) * delta;

				RecursiveWeightError->AdjustValue(recursive, neuron, wt);
			}

		}

		BiasError->AdjustValue(neuron, delta);
	}

	delete derivate;
	for (size_t i = 0; i < powers.size(); i++)
		delete powers[i];
	powers.clear();

	if (recursive)
		LayerInput->GetBackwardPass(LayerError);
}

void RecurrentLayer::Train(Optimizer* optimizer)
{
	optimizer->ModifyWeights(Weights, WeightError);
	optimizer->ModifyWeights(RecursiveWeight, RecursiveWeightError);
	optimizer->ModifyWeights(Bias, BiasError);
}

void RecurrentLayer::SetTrainingMode(bool mode)
{
	TrainingMode = mode;
	if (!mode)
	{
		while (!TrainingStates.empty())
			TrainingStates.pop_front();
	}
}

Matrix* RecurrentLayer::GetWeights()
{
	return Weights;
}

Matrix* RecurrentLayer::GetBias()
{
	return Bias;
}

Matrix* RecurrentLayer::GetRecurrentWeights()
{
	return RecursiveWeight;
}

