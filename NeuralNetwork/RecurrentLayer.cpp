#include "RecurrentLayer.h"

RecurrentLayer::RecurrentLayer(Layer* inputLayer, unsigned int size, unsigned int timeSteps) : Layer(inputLayer, size), TimeSteps(timeSteps), CurrentStep(0)
{
	Weights = new Matrix(inputLayer->GetSize(), size);
	Output = new Matrix(1, size);
	Bias = new Matrix(1, size);
	InnerState = new Matrix(1, size);
	WeightError = new Matrix(inputLayer->GetSize(), size);
	LayerError = new Matrix(1, inputLayer->GetSize());
	SavedState = new Matrix(1, size);
	//BiasError = new Matrix(1, size);
	RecursiveWeight = new Matrix(size, size);
	function = new TanhFunction();

	//MatrixMath::FillWith(Bias, 1);
}

RecurrentLayer::~RecurrentLayer()
{
	delete RecursiveWeight;
	delete function;
	delete SavedState;
}

void RecurrentLayer::Compute()
{
	/*Matrix* input = inputLayer->GetOutput();
	Matrix prevState(InnerState->GetRowCount(), InnerState->GetColumnCount());
	function->CalculateInto(InnerState, &prevState);
	MatrixMath::FillWith(InnerState, 0);
	MatrixMath::Multiply(input, Weights, InnerState);
	Matrix* timeState = MatrixMath::Multiply(&prevState, RecursiveWeight);
	MatrixMath::AddIn(InnerState, Bias);
	MatrixMath::AddIn(InnerState, timeState);
	function->CalculateInto(InnerState, Output);

	delete timeState;*/

	CurrentStep++;
	Matrix* input = inputLayer->GetOutput();


}

Matrix* RecurrentLayer::GetOutput()
{
	Compute();
	return Output;
}

void RecurrentLayer::GetBackwardPass(Matrix* error, bool recursive)
{
	//chain rule

}

