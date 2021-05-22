#include "GradientDescent.h"
#include "InputLayer.h"
#include "Constants.h"

#if DEBUG
#include "MatrixMath.h"
#endif // DEBUG


GradientDescent::GradientDescent(LossFuction loss, LossDerivate derivate, Layer* output, float learningRate) : Optimizer(output), LearningRate(learningRate)
{
	this->derivate = derivate;
	this->loss = loss;
}

GradientDescent::~GradientDescent()
{
}

Matrix GradientDescent::CalculateOutputError(const Matrix& output, const Matrix& expected)
{
	Matrix outputError(1, output.GetColumnCount());
	for (unsigned int i = 0; i < output.GetColumnCount(); i++)
	{
		outputError.SetValue(i, derivate(output, expected, i));
	}
	return outputError;
}

void GradientDescent::TrainStep(const Matrix& input, const Matrix& output)
{
	inputLayer->SetInput(input);
	Matrix outputValue = outputLayer->ComputeAndGetOutput();
	Matrix outputError = CalculateOutputError(outputValue, output);
	outputLayer->GetBackwardPass(outputError, true);
}

void GradientDescent::Train(const Matrix& input, const Matrix& expected)
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
	Matrix outputValue = outputLayer->ComputeAndGetOutput(); 
	//calculate errors
	Matrix outputError = CalculateOutputError(outputValue, expected);
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

}

void GradientDescent::ModifyWeights(Matrix& weights, const Matrix& errors)
{
	for (unsigned int row = 0; row < weights.GetRowCount(); row++)
	{
		for (unsigned int col = 0; col < weights.GetColumnCount(); col++)
		{
			float edit = -LearningRate * errors.GetValue(row, col) / currentBatch;
			weights.AdjustValue(row, col, edit);
		}
	}
}

void GradientDescent::Reset()
{
	outputLayer = nullptr;
	inputLayer = nullptr;
	currentBatch = 0;
}
