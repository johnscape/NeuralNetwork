#pragma once
#include "Optimizer.h"
class GradientDescent :
    public Optimizer
{
public:
    GradientDescent(LossFuction loss, LossDerivate derivate, Layer* output, float learningRate);
    virtual ~GradientDescent();

    virtual void Train(Matrix* input, Matrix* expected);

private:
    float LearningRate;
};

