#pragma once

#include "Matrix.h"

class Layer
{
public:
	Layer(Layer* inputLayer, unsigned int size);
	~Layer();

	virtual void SetInput(Layer* input);
	virtual void Compute() = 0;
	virtual Matrix* GetOutput();
	virtual unsigned int OutputSize();

	virtual unsigned int GetSize();
protected:
	Layer* inputLayer;
	Matrix* Output;

	unsigned int Size;

	static unsigned int Id;
};

