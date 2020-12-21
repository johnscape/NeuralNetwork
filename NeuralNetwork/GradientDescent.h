#pragma once
#include "Optimizer.h"
#include <memory>
class GradientDescent :
    public Optimizer
{
public:
    GradientDescent(LossFuction loss, LossDerivate derivate, Layer* output, float learningRate);
    virtual ~GradientDescent();

    virtual void Train(Matrix* input, Matrix* expected);
    virtual void ModifyWeights(Matrix* weights, Matrix* errors);
    virtual void Reset();

private:
    float LearningRate;

    LossFuction loss;
    LossDerivate derivate;

    Matrix* CalculateOutputError(Matrix* output, Matrix* expected);
    virtual void TrainStep(Matrix* input, Matrix* output);
};

