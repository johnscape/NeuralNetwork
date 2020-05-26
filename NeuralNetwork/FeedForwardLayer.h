#pragma once
#include "Layer.h"

#include "ActivationFunctions.hpp"

class FeedForwardLayer :
	public Layer
{
public:
	FeedForwardLayer(Layer* inputLayer, unsigned int count);
	~FeedForwardLayer();

	virtual void SetInput(Layer* input);
	virtual void Compute();
	virtual Matrix* GetOutput();

	void SetActivationFunction(ActivationFunction* func);

	Matrix* GetWeights();
	Matrix* GetBias();

private:
	Matrix* weights;
	Matrix* bias;
	Matrix* inner;
	unsigned int size;

	ActivationFunction* function;

};

