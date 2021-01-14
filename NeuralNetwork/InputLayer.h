#pragma once
#include "Layer.h"
class InputLayer :
	public Layer
{
public:
	InputLayer(unsigned int size);
	virtual ~InputLayer() {}
	virtual Layer* Clone();

	virtual void Compute();
	virtual Matrix* ComputeAndGetOutput();

	virtual void SetInput(Matrix* input);

	virtual void GetBackwardPass(Matrix* error, bool recursive = false);
	virtual void Train(Optimizer* optimizer) {}

	virtual void LoadFromJSON(const char* data, bool isFile = false);
	virtual std::string SaveToJSON(const char* fileName = nullptr);
private:
	unsigned int Size;
};

