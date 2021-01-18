#pragma once
#include "Optimizer.h"
#include <memory>
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
     * @param derivate The loss function's derivate.
     * @param output The output layer of the model.
     * @param learningRate The learning rate (alpha) of the optimizer.
    */
    GradientDescent(LossFuction loss, LossDerivate derivate, Layer* output, float learningRate);
    virtual ~GradientDescent();

    /**
     * @brief Trains the model based on the input and the expected output.
     * @param input The input of the model.
     * @param expected The expected output of the model.
    */
    virtual void Train(Matrix* input, Matrix* expected);

    /**
     * @brief Based on the type of the optimizer, this function will modify the weights of the layers.
     * @param weights The weights to modify
     * @param errors The error to calculate the new weights from
    */
    virtual void ModifyWeights(Matrix* weights, Matrix* errors);

    /**
     * @brief Resets the trainer and the network
    */
    virtual void Reset();

private:
    float LearningRate;

    LossFuction loss;
    LossDerivate derivate;

    Matrix* CalculateOutputError(Matrix* output, Matrix* expected);
    virtual void TrainStep(Matrix* input, Matrix* output);
};

