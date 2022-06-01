//
// Created by attila on 2022.06.01..
//

#ifndef NEURALNETWORK_LOSSFUNCTION_HPP
#define NEURALNETWORK_LOSSFUNCTION_HPP

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
#endif //NEURALNETWORK_LOSSFUNCTION_HPP
