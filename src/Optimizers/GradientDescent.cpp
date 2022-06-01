#include "NeuralNetwork/Optimizers/GradientDescent.h"
#include "NeuralNetwork/Layers/InputLayer.h"
#include "NeuralNetwork/Constants.h"

//TODO: Replace pointer
GradientDescent::GradientDescent(LossFunction* lossFunction, Layer* output, float learningRate) :
	Optimizer(output), LearningRate(learningRate)
{
	errorFunction = lossFunction;
}

GradientDescent::~GradientDescent()
{
	delete errorFunction;
}

Tensor GradientDescent::CalculateOutputError(const Tensor& output, const Tensor& expected)
{

	Tensor outputError(output.GetShape(), nullptr);
	for (unsigned int i = 0; i < outputError.GetElementCount(); i++)
		outputError.SetValue(i, errorFunction->Derivate(output, expected, i));
	return outputError;
}

void GradientDescent::TrainStep(const Tensor& input, const Tensor& output)
{
	inputLayer->SetInput(input);
	Tensor outputValue = outputLayer->ComputeAndGetOutput();
	Tensor outputError = CalculateOutputError(outputValue, output);
	outputLayer->GetBackwardPass(outputError, true);
}

void GradientDescent::Train(const Tensor& input, const Tensor& expected)
{
	currentBatch = input.GetShapeAt(0);
	//find input layer
	Layer* currentLayer = outputLayer;
	while (currentLayer->GetInputLayer())
	{
		currentLayer->SetTrainingMode(true);
		currentLayer = currentLayer->GetInputLayer();
	}
	//calculate
	currentLayer->SetInput(input);
	Tensor outputValue = outputLayer->ComputeAndGetOutput();
	//calculate errors
	Tensor outputError = CalculateOutputError(outputValue, expected);
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
			float edit = -LearningRate * errors.GetValue(row, col) / (float)currentBatch;
			weights.AdjustValue(row, col, edit);
		}
	}
}

void GradientDescent::Reset()
{
	outputLayer = nullptr;
	inputLayer = nullptr;
	currentBatch = 1;
}
