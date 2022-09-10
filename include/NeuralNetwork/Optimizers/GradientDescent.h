#pragma once
#include "Optimizer.h"
#include <memory>
#include "NeuralNetwork/LossFunctions/LossFunction.hpp"
/**
 * @brief Optimizer for supervised learning, using gradient descent
*/
class GradientDescent :
    public Optimizer
{
public:
    /**
     * @brief Creates a gradient descent optimizer.
     * @param loss The function which calculates the loss.
     * @param output The output layer of the model.
     * @param learningRate The learning rate (alpha) of the optimizer.
    */
    GradientDescent(LossFunction* lossFunction, Layer* output, float learningRate);

	/**
	 * @brief Creates a gradient descent optimizer.
	 * @param lossFunction The function which calculates the loss.
	 * @param model The model to train
	 * @param learningRate The learning rate (alpha) of the optimizer.
	 */
	GradientDescent(LossFunction* lossFunction, Model* model, float learningRate);
    virtual ~GradientDescent();

    /**
     * @brief Trains the model based on the input and the expected output.
     * @param input The input of the model.
     * @param output The expected output of the model.
    */
    virtual void TrainStep(const Tensor& input, const Tensor& output);

	virtual void Train(const Tensor& input, const Tensor& output, unsigned int batchDimension = 0);

    /**
     * @brief Based on the type of the optimizer, this function will modify the weights of the layers.
     * @param weights The weights to modify
     * @param errors The error to calculate the new weights from
    */
    virtual void ModifyWeights(Matrix& weights, const Matrix& errors);
	virtual void ModifyWeights(Tensor& weights, const Tensor& errors);
	
    /**
     * @brief Resets the trainer and the network
    */
    virtual void Reset();

private:
    float LearningRate;

    LossFunction* errorFunction;

    Tensor CalculateOutputError(const Tensor& output, const Tensor& expected);
};

