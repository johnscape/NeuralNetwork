#include "InputLayer.h"
#include "MatrixMath.h"
#include "LayerException.hpp"

InputLayer::InputLayer(unsigned int size) : Layer()
{
	LayerInput = nullptr;
	Output = new Matrix(1, size);
}

void InputLayer::Compute()
{
	return;
}

Matrix* InputLayer::ComputeAndGetOutput()
{
	return Output;
}

void InputLayer::SetInput(Matrix* input)
{
#if DEBUG
	if (!MatrixMath::SizeCheck(input, Output))
		return; //TODO: Throw error
#endif // DEBUG
	MatrixMath::Copy(input, Output);
}

void InputLayer::GetBackwardPass(Matrix* error, bool recursive)
{
}

