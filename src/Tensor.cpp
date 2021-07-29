#include "NeuralNetwork/Tensor.h"
#include "NeuralNetwork/TensorException.hpp"

Tensor::Tensor() : Values(nullptr), ElementCount(0)
{
}

Tensor::Tensor(unsigned int* dimensions, unsigned int dimensionCount, float* values) : ElementCount(1)
{
	if (dimensionCount == 0)
		dimensionCount = *(&dimensions + 1) - dimensions;

	for (unsigned int i = 0; i < dimensionCount; i++)
	{
		Shape.push_back(dimensions[i]);
		ElementCount *= dimensions[i];
	}

	Values = new float[ElementCount];

	if (values != nullptr)
		std::copy(values, values + ElementCount, Values);
	else
		std::fill(Values, Values + ElementCount, 0);
}

Tensor::Tensor(std::vector<unsigned int> dimensions, float* values) : ElementCount(1)
{
	Shape = dimensions;
	for (unsigned int i = 0; i < Shape.size(); i++)
		ElementCount *= Shape[i];

	Values = new float[ElementCount];

	if (values != nullptr)
		std::copy(values, values + ElementCount, Values);
	else
		std::fill(Values, Values + ElementCount, 0);
}

Tensor::Tensor(const Matrix& mat)
{
	ElementCount = mat.GetElementCount();
	Shape.push_back(mat.GetRowCount());
	Shape.push_back(mat.GetColumnCount());

	Values = new float[ElementCount];
	std::copy(mat.Values, mat.Values + ElementCount, Values);
}

Tensor::~Tensor()
{
	if (Values)
		delete[] Values;
}

bool Tensor::IsSameShape(const Tensor& other) const
{
	if (Shape.size() != other.Shape.size())
		return false;
	return Shape == other.Shape;
}

void Tensor::Reshape(std::vector<unsigned int> newDimensions)
{
	unsigned int sum1 = Shape[0];
	unsigned int sum2 = newDimensions[0];

	for (unsigned int i = 1; i < Shape.size(); i++)
		sum1 *= Shape[i];
	
	for (unsigned int i = 1; i < newDimensions.size(); i++)
		sum2 *= newDimensions[i];

	if (sum1 != sum2)
		throw TensorShapeException();

	Shape = newDimensions;
}

void Tensor::Reshape(unsigned int* newDimensions, unsigned int dimensionCount)
{
	if (dimensionCount == 0)
		dimensionCount = *(&newDimensions + 1) - newDimensions;
	unsigned int sum1 = Shape[0];
	unsigned int sum2 = newDimensions[0];

	for (unsigned int i = 1; i < Shape.size(); i++)
		sum1 *= Shape[i];

	for (unsigned int i = 1; i < dimensionCount; i++)
		sum2 *= newDimensions[i];

	if (sum1 != sum2)
		throw TensorShapeException();

	Shape.clear();
	Shape = std::vector<unsigned int>(newDimensions, newDimensions + dimensionCount);
}

std::vector<unsigned int> Tensor::GetShape() const
{
	return Shape;
}

std::string Tensor::GetShapeAsString() const
{
	std::string txt = "";
	for (unsigned int i = 0; i < Shape.size(); i++)
	{
		txt += Shape[i];
		if (i < Shape.size() - 1)
			txt += "x";
	}
	return txt;
}

Matrix Tensor::FirstMatrix() const
{
	if (Shape.size() >= 2)
	{
		Matrix mat(Shape[0], Shape[1]);
		std::copy(Values, Values + (Shape[0] * Shape[1]), mat.Values);
		return mat;
	}
	else if (Shape.size() == 1)
	{
		Matrix mat(Shape[0], 1);
		std::copy(Values, Values + Shape[0], mat.Values);
		return mat;
	}
	else
		return Matrix();
}

std::list<Matrix> Tensor::ToMatrixList() const
{
	std::list<Matrix> items;
	if (Shape.size() <= 2)
		items.push_back(FirstMatrix());
	else
	{
		unsigned int itemCount = Shape[2];
		if (Shape.size() >= 3)
			for (size_t i = 3; i < Shape.size(); i++)
				itemCount *= Shape[i];
		unsigned int elementCount = Shape[0] * Shape[1];
		
		for (size_t i = 0; i < itemCount; i++)
		{
			Matrix m(Shape[0], Shape[1]);
			std::copy(Values + i * elementCount, Values + (i + 1) * elementCount, m.Values);
			items.push_back(m);
		}
	}
	return items;
}