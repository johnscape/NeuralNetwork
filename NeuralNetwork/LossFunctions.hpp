#pragma once

#include "Matrix.h"
#include "MatrixMath.h"
#include "Constants.h"
#include "MatrixException.hpp"

namespace LossFunctions
{
	float MSE(std::shared_ptr<Matrix> output, std::shared_ptr<Matrix> expected)
	{
		float sum = 0;
		unsigned int elements = output->GetColumnCount() * output->GetRowCount();
		for (unsigned int i = 0; i < elements; i++)
			sum += pow(expected->GetValue(i) - output->GetValue(i), 2);
		return sum / (float)elements;
	}

	float MSE_Derivate(std::shared_ptr<Matrix> output, std::shared_ptr<Matrix> expected, unsigned int selected)
	{
		return output->GetValue(selected) - expected->GetValue(selected);
	}
}