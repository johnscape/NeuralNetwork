#pragma once

#include "NeuralNetwork/Tensor.h"

/**
 * @brief An abstract class for loss functions
 */
class LossFunction
{
public:
	virtual float Loss(const Tensor& output, const Tensor& expected) const = 0;
	virtual float Derivate(const Tensor& output, const Tensor& expected, unsigned int selected) const = 0;
};