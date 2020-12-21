#pragma once

#include "Matrix.h"
#include <memory>

class Optimizer;

class Layer
{
public:
	Layer(Layer* inputLayer);
	Layer();
	virtual ~Layer();

	virtual void SetInput(Layer* input);
	virtual void SetInput(Matrix* input) {}

	virtual void Compute() = 0;
	virtual Matrix* GetOutput();
	virtual unsigned int OutputSize();
	virtual Matrix* ComputeAndGetOutput() = 0;

	virtual Layer* GetInputLayer();

	virtual void GetBackwardPass(Matrix* error, bool recursive = false) = 0;
	virtual void Train(Optimizer* optimizer) = 0;
	virtual Matrix* GetLayerError();

	virtual void SetTrainingMode(bool mode, bool recursive = true);

protected:
	Layer* LayerInput;
	Matrix* Output;
	Matrix* LayerError;

	static unsigned int Id;
	bool TrainingMode;
};

