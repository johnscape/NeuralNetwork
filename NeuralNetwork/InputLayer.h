#pragma once
#include "Layer.h"
class InputLayer :
	public Layer
{
public:
	InputLayer(unsigned int size);
	virtual ~InputLayer() {}

	virtual void Compute();
	virtual Matrix* ComputeAndGetOutput();

	virtual void SetInput(Matrix* input);

	virtual void GetBackwardPass(Matrix* error, bool recursive = false);
	virtual void Train(Optimizer* optimizer) {}
};

