#pragma once
#include "Layer.h"

#include "ActivationFunctions.hpp"

class FeedForwardLayer :
	public Layer
{
public:
	FeedForwardLayer(Layer* inputLayer, unsigned int count);
	virtual Layer* Clone();
	virtual ~FeedForwardLayer();

	virtual void SetInput(Layer* input);
	virtual void Compute();
	virtual Matrix* ComputeAndGetOutput();

	void SetActivationFunction(ActivationFunction* func);

	virtual void GetBackwardPass(Matrix* error, bool recursive = false);

	virtual void Train(Optimizer* optimizer);

	Matrix* GetBias();
	Matrix* GetWeights();

private:

	ActivationFunction* function;


	Matrix* Weights;
	Matrix* Bias;
	Matrix* InnerState;

	Matrix* WeightError;
	Matrix* BiasError;

	unsigned int Size;
};

