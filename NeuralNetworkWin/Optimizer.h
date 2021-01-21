#pragma once

#include "Matrix.h"
#include "Layer.h"
#include <memory>

#include <vector>

typedef float (*LossFuction)(Matrix*, Matrix*);
typedef float (*LossDerivate)(Matrix*, Matrix*, unsigned int);

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
	 * @brief Trains the model based on the input and the expected output
	 * @param input The input of the model
	 * @param expected The expected output of the model
	*/
	[[Obsolete("Function is obsolete, use TrainStep or TrainFor instead")]]
	virtual void Train(Matrix* input, Matrix* expected) = 0;

	/**
	 * @brief Trains the model based on the input and the expected output
	 * @param input The input of the model
	 * @param expected The expected output of the model
	*/
	virtual void TrainStep(Matrix* input, Matrix* expected) = 0;

	/**
	 * @brief Based on the type of the optimizer, this function will modify the weights of the layers.
	 * @param weights The weights to modify
	 * @param errors The error to calculate the new weights from
	*/
	virtual void ModifyWeights(Matrix* weights, Matrix* errors) = 0;

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
	virtual void TrainFor(Matrix* input, Matrix* expected, unsigned int times, unsigned int batch = 32);

	/**
	 * @brief Runs the optimizer, until the error is below a specified value.
	 * @param input The input of the network. One row will be calculated at a time, giving multiple rows can be used as multiple samples.
	 * @param expected The expected values of the network, multiple rows can be given for multiple inputs.
	 * @param error The maximum acceptable error value
	 * @param batch The batch number, used for batch normalization
	*/
	[[deprecated("Not recommended, use TrainFor instead!")]]
	virtual void TrainUntil(Matrix* input, Matrix* expected, float error, unsigned int batch = 32);

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

};

