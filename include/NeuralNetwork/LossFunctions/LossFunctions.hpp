#pragma once

#include "NeuralNetwork/Tensor.h"
#include "NeuralNetwork/Constants.h"
#include "NeuralNetwork/MatrixException.hpp"

namespace LossFunctions
{
	inline float MSE(const Tensor& output, const Tensor& expected)
	{
		float sum = 0;
		for (int i = 0; i < output.GetElementCount(); ++i)
			sum += (float)pow(expected.GetValue(i) - output.GetValue(i), 2);
		return sum / (float)output.GetElementCount();
	}

	inline float MSE_Derivate(const Tensor& output, const Tensor& expected, unsigned int selected)
	{
		return output.GetValue(selected) - expected.GetValue(selected);
	}
}