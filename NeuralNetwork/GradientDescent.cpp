#include "GradientDescent.h"
#include "InputLayer.h"

//TODO: Delete
#include "MatrixMath.h"
#include <iostream>

GradientDescent::GradientDescent(LossFuction loss, LossDerivate derivate, Layer* output, float learningRate) : Optimizer(output), LearningRate(learningRate)
{
	this->derivate = derivate;
	this->loss = loss;
}

GradientDescent::~GradientDescent()
{
}

Matrix* GradientDescent::CalculateOutputError(Matrix* output, Matrix* expected)
{
	Matrix* outputError(new Matrix(1, output->GetColumnCount()));
	for (unsigned int i = 0; i < output->GetColumnCount(); i++)
	{
		outputError->SetValue(i, derivate(output, expected, i));
	}
	return outputError;
}

void GradientDescent::TrainStep(Matrix* input, Matrix* output)
{
	inputLayer->SetInput(input);
	Matrix* outputValue = outputLayer->ComputeAndGetOutput();
	Matrix* outputError = CalculateOutputError(outputValue, output);
	outputLayer->GetBackwardPass(outputError, true);
	delete outputError;
}

void GradientDescent::Train(Matrix* input, Matrix* expected)
{
	//find input layer
	Layer* currentLayer = outputLayer;
	while (currentLayer->GetInputLayer())
	{
		currentLayer->SetTrainingMode(true);
		currentLayer = currentLayer->GetInputLayer();
	}
	//calculate
	currentLayer->SetInput(input);
	Matrix* outputValue = outputLayer->ComputeAndGetOutput();
	//calculate errors
	Matrix* outputError = CalculateOutputError(outputValue, expected);
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

	delete outputError;

}

void GradientDescent::ModifyWeights(Matrix* weights, Matrix* errors)
{
	//MatrixMath::PrintMatrix(errors);
	for (unsigned int row = 0; row < weights->GetRowCount(); row++)
	{
		for (unsigned int col = 0; col < weights->GetColumnCount(); col++)
		{
			float edit = -LearningRate * errors->GetValue(row, col) / 1; //TODO: set to current batch
			weights->AdjustValue(row, col, edit);
		}
	}
}

void GradientDescent::Reset()
{
	outputLayer = nullptr;
	inputLayer = nullptr;
	currentBatch = 0;
}
