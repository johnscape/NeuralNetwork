#include "NeuralNetwork/Layers/ConvLayer.h"

ConvLayer::ConvLayer(Layer* inputLayer, unsigned int kernelSize, unsigned int stride, unsigned int nrFilters,
	 unsigned int minimumPad, Matrix::PadType padType, float padFill) : Layer(inputLayer), Stride(stride),
	 PadFill(padFill), PaddingType(padType)
{
	std::vector<unsigned int> outputShape = inputLayer->GetOutput().GetShape();
	unsigned int firstDim = outputShape[0] - kernelSize;
	unsigned int secondDim = outputShape[1] - kernelSize;
	unsigned int pad = minimumPad;
	while ((firstDim + 2 * pad) % (stride + 1) != 0 && (secondDim + 2 * pad) % (stride + 1) != 0)
		pad++;
	PadSize = pad;
	outputShape[0] = (firstDim + 2 * PadSize) / (Stride + 1);
	outputShape[1] = (secondDim + 2 * PadSize) / (Stride + 1);
	outputShape[2] = nrFilters;

	Kernel = Tensor({kernelSize, kernelSize, inputLayer->GetOutput().GetShapeAt(2), nrFilters});
	Kernel.FillWithRandom();

	Output = Tensor(outputShape);
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

}

Tensor &ConvLayer::ComputeAndGetOutput()
{
	Compute();
	return Output;
}

void ConvLayer::GetBackwardPass(const Tensor &error, bool recursive)
{

}

void ConvLayer::Train(Optimizer *optimizer)
{

}

void ConvLayer::LoadFromJSON(const char *data, bool isFile)
{

}

std::string ConvLayer::SaveToJSON(const char *fileName)
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
