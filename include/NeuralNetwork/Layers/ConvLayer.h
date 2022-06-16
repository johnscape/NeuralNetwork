#pragma once

#include "Layer.h"

class ActivationFunction;

class ConvLayer : public Layer
{
public:
	ConvLayer(Layer* inputLayer, unsigned int kernelSize, unsigned int stride, unsigned int nrFilters = 1,
			  unsigned int minimumPad = 0, Matrix::PadType padType = Matrix::PadType::CONSTANT, float padFill = 0);
	virtual ~ConvLayer();

	virtual Layer* Clone();
	virtual void Compute();
	virtual Tensor& ComputeAndGetOutput();
	virtual void GetBackwardPass(const Tensor& error, bool recursive = false);
	virtual void Train(Optimizer* optimizer);
	virtual void LoadFromJSON(const char* data, bool isFile = false);
	virtual std::string SaveToJSON(const char* fileName = nullptr);

	Tensor& GetKernel();
	unsigned int GetPadSize();


private:
	Tensor Kernel;
	unsigned int PadSize;
	unsigned int Stride;
	Matrix::PadType PaddingType;
	float PadFill;

	Tensor KernelError;

	ActivationFunction* function;
};
