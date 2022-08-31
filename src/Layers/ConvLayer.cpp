#include "NeuralNetwork/Layers/ConvLayer.h"
#include "NeuralNetwork/ActivationFunctions.hpp"
#include "NeuralNetwork/TensorException.hpp"
#include "NeuralNetwork/Optimizers/Optimizer.h"
#include <cmath>

ConvLayer::ConvLayer(Layer* inputLayer, unsigned int kernelSize, unsigned int stride, unsigned int nrFilters,
	 unsigned int minimumPad, Matrix::PadType padType, float padFill) : Layer(inputLayer), Stride(stride),
	 PadFill(padFill), PaddingType(padType)
{
	std::vector<unsigned int> outputShape = inputLayer->GetOutput().GetShape();
	unsigned int firstDim = outputShape[0] - kernelSize;
	unsigned int secondDim = outputShape[1] - kernelSize;
	PadSize = minimumPad;
	if (Stride > 1)
	{
		float padFirst = (float)((Stride - 1) * firstDim - Stride + kernelSize) / 2;
		float padSecond= (float)((Stride - 1) * secondDim - Stride + kernelSize) / 2;
		if (padFirst > PadSize || padSecond > PadSize)
			PadSize = padFirst > padSecond ? ceil(padFirst) : ceil(padSecond);
	}
	outputShape[0] = ((firstDim + 2 * PadSize) / Stride) + 1;
	outputShape[1] = ((secondDim + 2 * PadSize) / Stride) + 1;
	outputShape[2] = nrFilters;

	//TODO: Handle different dimensions
	//Kernel shape: rows	cols	inputChannels	nrFilters
	Kernel = Tensor({kernelSize, kernelSize, inputLayer->GetOutput().GetShapeAt(2), nrFilters});
	Kernel.FillWithRandom();

	Output = Tensor(outputShape);
	function = &RELU::GetInstance();

	LayerError = Tensor(inputLayer->GetOutput().GetShape(), nullptr);
	KernelError = Tensor(Kernel.GetShape(), nullptr);
}

ConvLayer::~ConvLayer()
{

}

Layer *ConvLayer::Clone()
{
	return nullptr;
}

void ConvLayer::Compute()
{
	Tensor input = LayerInput->ComputeAndGetOutput();
	if (TrainingMode) //Set up training input
	{
		if (PadSize == 0)
			InputStep = input;
		else
		{
			std::vector<unsigned int> shape = input.GetShape();
			shape[0] += 2 * PadSize;
			shape[1] += 2 * PadSize;
			InputStep = Tensor(shape, nullptr);
		}
	}
	unsigned int kernelRowCount = Kernel.GetShapeAt(0);
	unsigned int channelRowCount = kernelRowCount * Kernel.GetShapeAt(2);

	unsigned int inputSteps = input.GetMatrixCount() / Kernel.GetShapeAt(2);
	Matrix convResult(Output.GetShapeAt(0), Output.GetShapeAt(1));

	Matrix inputStep(input.GetShapeAt(0), input.GetShapeAt(1));

	for (int is = 0; is < inputSteps; ++is)
	{
		for (unsigned int k = 0; k < Kernel.GetShapeAt(3); k++) //kernel selection
		{
			for (unsigned int ch = 0; ch < Kernel.GetShapeAt(2); ++ch) //channel selection
			{
				inputStep.Reset(input.GetShapeAt(0), input.GetShapeAt(1));
				TempMatrix ker = Kernel.GetNthTempMatrix(ch + k * Kernel.GetShapeAt(2));
				//TempMatrix inp = input.GetNthTempMatrix(ch + Kernel.GetShapeAt(2) * is);
				input.GetNthMatrix(ch + Kernel.GetShapeAt(2) * is, &inputStep);
				if (PadSize > 0)
				{
					inputStep.Pad(PadSize, PadSize, PadSize, PadSize, PaddingType, PadFill);
					if (TrainingMode)
						InputStep.LoadMatrix(ch + Kernel.GetShapeAt(2) * is, &inputStep);
				}
				inputStep.Convolute(ker, Stride, &convResult);

				TempMatrix currentOutput = Output.GetNthTempMatrix(k + Kernel.GetShapeAt(3) * is);
				convResult += currentOutput;
				Output.LoadMatrix(k + Kernel.GetShapeAt(3) * is, &convResult);
			}
		}
	}
	function->CalculateInto(Output, Output);
}

Tensor &ConvLayer::ComputeAndGetOutput()
{
	Compute();
	return Output;
}

void ConvLayer::GetBackwardPass(const Tensor &error, bool recursive)
{
	KernelError.FillWith(0);
	LayerError.FillWith(0);

	//Kernel error
	Tensor derivate = function->CalculateDerivateTensor(error);

	Matrix inputStep(InputStep.GetShapeAt(0), InputStep.GetShapeAt(1));
	Matrix convResult(Kernel.GetShapeAt(0), Kernel.GetShapeAt(1));

	for (unsigned int c = 0; c < Output.GetShapeAt(2); c++)
	{
		for (unsigned int i = 0; i < InputStep.GetMatrixCount(); i++)
		{
			unsigned int inputChannel = i % InputStep.GetShapeAt(2);
			InputStep.GetNthMatrix(i, &inputStep);
			TempMatrix usedError = error.GetNthTempMatrix(c);
			inputStep.Convolute(usedError, Stride, &convResult);
			convResult += KernelError.GetNthTempMatrix(c * Kernel.GetShapeAt(2) + inputChannel);
			KernelError.LoadMatrix(c * Kernel.GetShapeAt(2) + inputChannel, &convResult);
		}
	}

	//Layer error
	Matrix tmpKernel(Kernel.GetShapeAt(0), Kernel.GetShapeAt(1));
	Matrix fullError(error.GetShapeAt(0) + 2, error.GetShapeAt(1) + 2);
	convResult.Reset(InputStep.GetShapeAt(0), InputStep.GetShapeAt(1));

	for (unsigned int k = 0; k < Kernel.GetShapeAt(3); ++k)
	{
		for (unsigned int i = 0; i < Kernel.GetShapeAt(2); ++i)
		{

			Kernel.GetNthMatrix(k * Kernel.GetShapeAt(2) + i, &tmpKernel);
			tmpKernel.Rotate(2);

			TempMatrix usedError = error.GetNthTempMatrix(k);
			usedError.Pad(1, 1, 1, 1, Matrix::PadType::CONSTANT, 0, &fullError);
			fullError.Convolute(tmpKernel, Stride, &convResult);
			convResult += LayerError.GetNthTempMatrix(i);
			LayerError.LoadMatrix(i, &convResult);

		}
	}

	if (recursive)
		LayerInput->GetBackwardPass(LayerError, true);
}

void ConvLayer::Train(Optimizer *optimizer)
{
	optimizer->ModifyWeights(Kernel, KernelError);
}

void ConvLayer::LoadFromJSON(const char *data, bool isFile)
{

}

std::string ConvLayer::SaveToJSON(const char *fileName) const
{
	return std::string();
}

Tensor &ConvLayer::GetKernel()
{
	return Kernel;
}

unsigned int ConvLayer::GetPadSize()
{
	return PadSize;
}
