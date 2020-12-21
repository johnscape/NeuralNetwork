#pragma once

#include "Matrix.h"
#include "Layer.h"
#include <memory>

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
	virtual void Reset() = 0;
	virtual void TrainFor(Matrix* input, Matrix* expected, unsigned int times, unsigned int batch = 32);
	virtual void TrainUntil(Matrix* input, Matrix* expected, float error, unsigned int batch = 32);

protected:
	Layer* outputLayer;
	Layer* inputLayer;

	float lastError;
	unsigned int currentBatch;

	virtual void FindInputLayer();
	virtual void TrainStep(Matrix* input, Matrix* expected) = 0;
	virtual void TrainLayers();

};

