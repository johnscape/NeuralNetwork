#pragma once

#include "Matrix.h"
#include "Layer.h"

#include <vector>

typedef float (*LossFuction)(Matrix*, Matrix*);
typedef float (*LossDerivate)(Matrix*, Matrix*, unsigned int);

class Optimizer
{
public:
	Optimizer(Layer* output);
	virtual ~Optimizer();

	virtual void Train(Matrix* input, Matrix* expected) = 0;
	virtual void ModifyWeights(Matrix* weights, Matrix* errors) = 0;

protected:
	Layer* outputLayer;

	
};

