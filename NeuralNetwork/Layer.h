#pragma once

#include "Matrix.h"

class Optimizer;

class Layer
{
public:
	Layer(Layer* inputLayer, unsigned int size);
	~Layer();

	virtual void SetInput(Layer* input);
	virtual void SetInput(Matrix* input) {}

	virtual void Compute() = 0;
	virtual Matrix* GetOutput();
	virtual unsigned int OutputSize();

	virtual unsigned int GetSize();

	virtual Layer* GetInputLayer();

	//virtual void Freeze();
	//virtual void Unfreeze();

	//virtual Matrix* CalculateErrors(Matrix* error);

	virtual Matrix* GetWeights();
	virtual Matrix* GetBias();
	virtual Matrix* GetInnerState();

	virtual Matrix* GetFrozenOutput();

	virtual void GetBackwardPass(Matrix* error, bool recursive = false) = 0;

	virtual Matrix* GetLayerError();

	virtual void Train(Optimizer* optimizer) = 0;

protected:
	Layer* inputLayer;
	Matrix* Output;

	Matrix* Weights;
	Matrix* InnerState;
	Matrix* Bias;

	Matrix* WeightError;
	Matrix* LayerError;

	unsigned int Size;

	static unsigned int Id;

	bool Frozen;
};

