#pragma once
#include "Optimizer.h"
#include <memory>
class GradientDescent :
    public Optimizer
{
public:
    GradientDescent(LossFuction loss, LossDerivate derivate, Layer& output, float learningRate);
    virtual ~GradientDescent();

    virtual void Train(Matrix& input, Matrix& expected);
    virtual void ModifyWeights(std::shared_ptr<Matrix> weights, std::shared_ptr<Matrix> errors);

private:
    float LearningRate;

    LossFuction loss;
    LossDerivate derivate;

    std::shared_ptr<Matrix> CalculateOutputError(std::shared_ptr<Matrix> output, std::shared_ptr<Matrix> expected);
};

