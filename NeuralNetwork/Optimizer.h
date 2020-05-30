#pragma once

#include "Matrix.h"
#include "Layer.h"

#include <vector>

typedef float (*LossFuction)(Matrix*, Matrix*);
typedef float (*LossDerivate)(Matrix*, Matrix*, unsigned int);

class Optimizer
{
public:
	Optimizer(LossFuction loss, LossDerivate derivate, Layer* output);
	virtual ~Optimizer();

	virtual void Train(Matrix* input, Matrix* expected) = 0;

protected:
	
	LossFuction loss;
	LossDerivate derivate;

	std::vector<Matrix*> errors;
	Layer* outputLayer;

	virtual void CalculateErrors(Layer* currentLayer);
	virtual void CalculateOutputError(Matrix* output, Matrix* expected);

	void ClearErrors();
};

