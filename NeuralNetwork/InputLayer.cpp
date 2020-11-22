#include "InputLayer.h"
#include "MatrixMath.h"
#include "LayerException.hpp"

InputLayer::InputLayer(unsigned int size) : Layer(*this)
{
	LayerInput.reset();
	Output.reset(new Matrix(1, size));
}

void InputLayer::Compute()
{
	return;
}

std::shared_ptr<Matrix> InputLayer::ComputeAndGetOutput()
{
	return Output;
}

void InputLayer::SetInput(std::shared_ptr<Matrix> input)
{
#if DEBUG
	if (!MatrixMath::SizeCheck(input, Output))
		return; //TODO: Throw error
#endif // DEBUG
	MatrixMath::Copy(input, Output);
}

void InputLayer::GetBackwardPass(std::shared_ptr<Matrix> error, bool recursive)
{
	throw LayerInputException();
}

