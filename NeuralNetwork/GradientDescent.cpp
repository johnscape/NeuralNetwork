#include "GradientDescent.h"
#include "InputLayer.h"

//TODO: Delete
#include "MatrixMath.h"
#include <iostream>

GradientDescent::GradientDescent(LossFuction loss, LossDerivate derivate, std::shared_ptr<Layer> output, float learningRate) : Optimizer(output), LearningRate(learningRate)
{
	this->derivate = derivate;
	this->loss = loss;
}

GradientDescent::~GradientDescent()
{
}

std::shared_ptr<Matrix> GradientDescent::CalculateOutputError(std::shared_ptr<Matrix> output, std::shared_ptr<Matrix> expected)
{
	std::shared_ptr<Matrix> outputError(new Matrix(1, output->GetColumnCount()));
	for (unsigned int i = 0; i < output->GetColumnCount(); i++)
	{
		outputError->SetValue(i, derivate(output, expected, i));
	}
	return outputError;
}

void GradientDescent::Train(std::shared_ptr<Matrix> input, std::shared_ptr<Matrix> expected)
{
	//find input layer
	std::shared_ptr<Layer> currentLayer = outputLayer;
	while (currentLayer->GetInputLayer())
	{
		currentLayer->SetTrainingMode(true);
		currentLayer = currentLayer->GetInputLayer();
	}
	//calculate
	currentLayer->SetInput(input);
	std::shared_ptr<Matrix> outputValue = outputLayer->ComputeAndGetOutput();
	//calculate errors
	std::shared_ptr<Matrix> outputError = CalculateOutputError(outputValue, expected);
	outputLayer->GetBackwardPass(outputError, true);

	currentLayer = outputLayer;
	while (currentLayer->GetInputLayer())
	{
		currentLayer->Train(this);
		currentLayer = currentLayer->GetInputLayer();
	}

	currentLayer = outputLayer;
	while (currentLayer->GetInputLayer())
	{
		currentLayer->SetTrainingMode(false);
		currentLayer = currentLayer->GetInputLayer();
	}

	outputError.reset();

}

void GradientDescent::ModifyWeights(std::shared_ptr<Matrix> weights, std::shared_ptr<Matrix> errors)
{
	for (unsigned int row = 0; row < weights->GetRowCount(); row++)
	{
		for (unsigned int col = 0; col < weights->GetColumnCount(); col++)
		{
			float edit = -LearningRate * errors->GetValue(row, col);
			weights->AdjustValue(row, col, edit);
		}
	}
}
