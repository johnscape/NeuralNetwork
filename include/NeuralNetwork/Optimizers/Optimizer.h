#pragma once

#include "NeuralNetwork/Tensor.h"
#include "NeuralNetwork/Layers/Layer.h"
#include <memory>

#include <vector>

//TODO: Use model class

/**
 * @brief This class is used to train the neural network.
*/
class Optimizer
{
public:
	/**
	 * @brief Creates an optimizer
	 * @param output The model's output layer
	*/
	Optimizer(Layer* output);
	virtual ~Optimizer();

	/**
	 * @brief Executes a single train step on the model based on the input and the expected output
	 * @param input The input of the model
	 * @param expected The expected output of the model
	*/
	virtual void Train(const Tensor& input, const Tensor& expected) = 0;

	/**
	 * @brief Trains the model based on the input and the expected output
	 * @param input The input of the model
	 * @param expected The expected output of the model
	*/
	virtual void TrainStep(const Tensor& input, const Tensor& expected) = 0;

	/**
	 * @brief Based on the type of the optimizer, this function will modify the weights of the layers.
	 * @param weights The weights to modify
	 * @param errors The error to calculate the new weights from
	*/
	virtual void ModifyWeights(Matrix& weights, const Matrix& errors) = 0;

	/**
	 * @brief Resets the trainer and the network
	*/
	virtual void Reset() = 0;

	/**
	 * @brief Trains the network for a specified number of steps.
	 * @param input The input of the network. One row will be calculated at a time, giving multiple rows can be used as multiple samples.
	 * @param expected The expected values of the network, multiple rows can be given for multiple inputs.
	 * @param times How many times the training step will be ran
	 * @param batch The batch number, used for batch normalization
	*/
	virtual void TrainFor(const Tensor& input, const Tensor& expected, unsigned int times, unsigned int batch = 32);

	/**
	 * @brief Runs the optimizer, until the error is below a specified value.
	 * @param input The input of the network. One row will be calculated at a time, giving multiple rows can be used as multiple samples.
	 * @param expected The expected values of the network, multiple rows can be given for multiple inputs.
	 * @param error The maximum acceptable error value
	 * @param batch The batch number, used for batch normalization
	*/
	[[deprecated("Not recommended, use TrainFor instead!")]]
	virtual void TrainUntil(const Matrix& input, const Matrix& expected, float error, unsigned int batch = 32);

	/**
	 * @brief Sets the training mode for the model.
	 * @param mode The value of the expected training mode.
	*/
	virtual void SetTrainingMode(bool mode);

protected:
	Layer* outputLayer;
	Layer* inputLayer;

	float lastError;
	unsigned int currentBatch;

	virtual void FindInputLayer();
	virtual void TrainLayers();

	Tensor GetBatch(const Tensor& original, unsigned int batchSize, unsigned int count);

};

