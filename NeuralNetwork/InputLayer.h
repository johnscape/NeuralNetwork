#pragma once
#include "Layer.h"
class InputLayer :
	public Layer
{
public:
	InputLayer(unsigned int size);
	~InputLayer() {}

	virtual void Compute();
	virtual std::shared_ptr<Matrix> ComputeAndGetOutput();

	virtual void SetInput(std::shared_ptr<Matrix> input);

	virtual void GetBackwardPass(std::shared_ptr<Matrix> error, bool recursive = false);
	virtual void Train(Optimizer* optimizer) {}
};

