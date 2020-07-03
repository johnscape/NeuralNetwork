#pragma once
#include "Layer.h"
#include "ActivationFunctions.hpp"
class RecurrentLayer :
    public Layer
{
	RecurrentLayer(Layer* inputLayer, unsigned int size, unsigned int timeSteps = 3);
	virtual ~RecurrentLayer();


	virtual void Compute();
	virtual Matrix* GetOutput();

	virtual void GetBackwardPass(Matrix* error, bool recursive = false);

	virtual void Train(Optimizer* optimizer);
private:
	unsigned int TimeSteps;
	Matrix* RecursiveWeight;
	ActivationFunction* function;

	Matrix* SavedState;
	unsigned int CurrentStep;
};

