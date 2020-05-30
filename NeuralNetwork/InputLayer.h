#pragma once
#include "Layer.h"
class InputLayer :
	public Layer
{
public:
	InputLayer(unsigned int size);
	~InputLayer() {}

	virtual void Compute();

	virtual void SetInput(Matrix* input);

	virtual void GetBackwardPass(Matrix* error);
};

