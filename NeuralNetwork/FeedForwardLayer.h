#pragma once
#include "Layer.h"

#include "ActivationFunctions.hpp"

class FeedForwardLayer :
	public Layer
{
public:
	FeedForwardLayer(Layer* inputLayer, unsigned int count);
	virtual ~FeedForwardLayer();

	virtual void SetInput(Layer* input);
	virtual void Compute();
	virtual Matrix* GetOutput();

	void SetActivationFunction(ActivationFunction* func);

	virtual void GetBackwardPass(Matrix* error, bool recursive = false);

	virtual void Train(Optimizer* optimizer);

private:

	ActivationFunction* function;

	Matrix* BiasError;

};

