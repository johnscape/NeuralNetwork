#pragma once

#include "Matrix.h"
#include "Layer.h"
#include <memory>

#include <vector>

typedef float (*LossFuction)(std::shared_ptr<Matrix>, std::shared_ptr<Matrix>);
typedef float (*LossDerivate)(std::shared_ptr<Matrix>, std::shared_ptr<Matrix>, unsigned int);

class Optimizer
{
public:
	Optimizer(std::shared_ptr<Layer> output);
	virtual ~Optimizer();

	virtual void Train(std::shared_ptr<Matrix> input, std::shared_ptr<Matrix> expected) = 0;
	virtual void ModifyWeights(std::shared_ptr<Matrix> weights, std::shared_ptr<Matrix> errors) = 0;

protected:
	std::shared_ptr<Layer> outputLayer;
};

