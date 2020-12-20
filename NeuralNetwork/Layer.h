#pragma once

#include "Matrix.h"
#include <memory>

class Optimizer;

class Layer
{
public:
	Layer(std::shared_ptr<Layer> inputLayer);
	Layer(Layer& inputLayer);
	Layer();
	~Layer();

	virtual void SetInput(std::shared_ptr<Layer> input);
	virtual void SetInput(std::shared_ptr<Matrix> input) {}

	virtual void Compute() = 0;
	virtual std::shared_ptr<Matrix> GetOutput();
	virtual unsigned int OutputSize();
	virtual std::shared_ptr<Matrix> ComputeAndGetOutput() = 0;

	virtual std::shared_ptr<Layer> GetInputLayer();

	virtual void GetBackwardPass(std::shared_ptr<Matrix> error, bool recursive = false) = 0;
	virtual void Train(Optimizer* optimizer) = 0;
	virtual std::shared_ptr<Matrix> GetLayerError();

	virtual void SetTrainingMode(bool mode, bool recursive = true);

protected:
	std::shared_ptr<Layer> LayerInput;
	std::shared_ptr<Matrix> Output;
	std::shared_ptr<Matrix> LayerError;

	static unsigned int Id;
	bool TrainingMode;

	//bool Frozen;
};

