#pragma once
#include "Layer.h"
#include "ActivationFunctions.hpp"
#include <queue>
#include <memory>
#include "Optimizer.h"

class RecurrentLayer : public Layer
{
public:
	RecurrentLayer(Layer* inputLayer, unsigned int size, unsigned int timeSteps = 3);
	virtual ~RecurrentLayer();
	virtual Layer* Clone();


	virtual void Compute();
	virtual Matrix* GetOutput();
	virtual Matrix* ComputeAndGetOutput();
	virtual void SetActivationFunction(ActivationFunction* func);

	virtual void GetBackwardPass(Matrix* error, bool recursive = false);

	virtual void Train(Optimizer* optimizer);
	virtual void SetTrainingMode(bool mode);

	Matrix* GetWeights();
	Matrix* GetBias();
	Matrix* GetRecurrentWeights();
private:
	unsigned int TimeSteps;
	ActivationFunction* function;

	unsigned int CurrentStep;
	unsigned int Size;

	Matrix* Weights;
	Matrix* Bias;
	Matrix* RecursiveWeight;
	Matrix* InnerState;

	Matrix* WeightError;
	Matrix* BiasError;
	Matrix* RecursiveWeightError;

	std::deque<Matrix*> TrainingStates;
	std::deque<Matrix*> IncomingValues;

};

