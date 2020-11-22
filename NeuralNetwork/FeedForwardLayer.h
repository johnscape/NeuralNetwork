#pragma once
#include "Layer.h"

#include "ActivationFunctions.hpp"

class FeedForwardLayer :
	public Layer
{
public:
	FeedForwardLayer(std::shared_ptr<Layer> inputLayer, unsigned int count);
	virtual ~FeedForwardLayer();

	virtual void SetInput(std::shared_ptr<Layer> input);
	virtual void Compute();
	virtual std::shared_ptr<Matrix> ComputeAndGetOutput();

	void SetActivationFunction(std::shared_ptr<ActivationFunction> func);

	virtual void GetBackwardPass(std::shared_ptr<Matrix> error, bool recursive = false);

	virtual void Train(Optimizer* optimizer);

	std::shared_ptr<Matrix> GetBias();
	std::shared_ptr<Matrix> GetWeights();

private:

	std::shared_ptr<ActivationFunction> function;


	std::shared_ptr<Matrix> Weights;
	std::shared_ptr<Matrix> Bias;
	std::shared_ptr<Matrix> InnerState;

	std::shared_ptr<Matrix> WeightError;
	std::shared_ptr<Matrix> BiasError;

	unsigned int Size;
};

