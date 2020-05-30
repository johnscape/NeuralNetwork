#include "GradientDescent.h"
#include "InputLayer.h"

//TODO: Delete
#include "MatrixMath.h"
#include <iostream>

GradientDescent::GradientDescent(LossFuction loss, LossDerivate derivate, Layer* output, float learningRate) : Optimizer(loss, derivate, output), LearningRate(learningRate)
{
}

GradientDescent::~GradientDescent()
{
}

void GradientDescent::Train(Matrix* input, Matrix* expected)
{
	//find input layer
	Layer* currentLayer = outputLayer;
	while (currentLayer->GetInputLayer())
		currentLayer = currentLayer->GetInputLayer();
	//calculate
	currentLayer->SetInput(input);
	Matrix* outputValue = outputLayer->GetOutput();
	//calculate errors
	CalculateOutputError(outputValue, expected);
	//go trough layers
	currentLayer = outputLayer;

	std::vector<Matrix*> weight_errors;
	
	while (currentLayer->GetInputLayer())
	{
		// Error/Output
		Matrix* prevError = errors[errors.size() - 1];
		// Layer inner
		currentLayer->GetBackwardPass(prevError);
		currentLayer = currentLayer->GetInputLayer();
	}
}
