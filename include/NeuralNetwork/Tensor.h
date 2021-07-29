#pragma once

#include <vector>
#include <list>
#include <string>
#include "NeuralNetwork/Matrix.h"

class Tensor
{
public:
	Tensor();
	Tensor(unsigned int* dimensions, unsigned int dimensionCount = 0, float* values = nullptr);
	Tensor(std::vector<unsigned int> dimensions, float* values = nullptr);
	Tensor(const Matrix& mat);

	~Tensor();

	bool IsSameShape(const Tensor& other) const;

	void Reshape(std::vector<unsigned int> newDimensions);
	void Reshape(unsigned int* newDimensions, unsigned int dimensionCount = 0);

	std::vector<unsigned int> GetShape() const;
	std::string GetShapeAsString() const;

	Matrix FirstMatrix() const;
	std::list<Matrix> ToMatrixList() const;
private:
	std::vector<unsigned int> Shape;
	float* Values;
	size_t ElementCount;
};