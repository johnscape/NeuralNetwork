#pragma once
#include "Layer.h"
class InputLayer :
	public Layer
{
public:
	InputLayer(unsigned int size);
	~InputLayer() {}

	virtual void Compute();

	void SetInput(Matrix* input);
};

