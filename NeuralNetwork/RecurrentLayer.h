#pragma once
#include "Layer.h"
#include "ActivationFunctions.hpp"
#include <queue>
#include <memory>

class RecurrentLayer : public Layer
{
public:
	RecurrentLayer(std::shared_ptr<Layer> inputLayer, unsigned int size, unsigned int timeSteps = 3);
	virtual ~RecurrentLayer();


	virtual void Compute();
	virtual std::shared_ptr<Matrix> GetOutput();
	virtual std::shared_ptr<Matrix> ComputeAndGetOutput();

	virtual void GetBackwardPass(std::shared_ptr<Matrix> error, bool recursive = false);

	virtual void Train(Optimizer* optimizer);
	virtual void SetTrainingMode(bool mode);
private:
	unsigned int TimeSteps;
	ActivationFunction* function;

	std::shared_ptr<Matrix> SavedState;
	unsigned int CurrentStep;
	unsigned int Size;

	std::shared_ptr<Matrix> Weights;
	std::shared_ptr<Matrix> Bias;
	std::shared_ptr<Matrix> RecursiveWeight;
	std::shared_ptr<Matrix> InnerState;

	std::shared_ptr<Matrix> WeightError;
	std::shared_ptr<Matrix> BiasError;
	std::shared_ptr<Matrix> RecursiveWeightError;

	std::queue<std::shared_ptr<Matrix>> TrainingStates;

};

