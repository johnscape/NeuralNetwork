//
// Created by attila on 2022.06.01..
//

#ifndef NEURALNETWORK_MSE_HPP
#define NEURALNETWORK_MSE_HPP

#include "NeuralNetwork/LossFunctions/LossFunction.hpp"
#include <cmath>

class MSE : public LossFunction
{
public:
	virtual float Loss(const Tensor& output, const Tensor& expected) const
	{
		float sum = 0;
		for (int i = 0; i < output.GetElementCount(); ++i)
			sum += (float)pow(expected.GetValue(i) - output.GetValue(i), 2);
		return sum / (float)output.GetElementCount();
	}

	virtual float Derivate(const Tensor& output, const Tensor& expected, unsigned int selected) const
	{
		return output.GetValue(selected) - expected.GetValue(selected);
	}
};
#endif //NEURALNETWORK_MSE_HPP
