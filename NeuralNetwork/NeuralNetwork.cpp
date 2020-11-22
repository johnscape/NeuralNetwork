// NeuralNetwork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "Matrix.h"
#include "InputLayer.h"
#include "FeedForwardLayer.h"
#include "RecurrentLayer.h"
#include "ActivationFunctions.hpp"
#include "Constants.h"

#include "GradientDescent.h"
#include "LossFunctions.hpp"

#include <memory>
int main()
{
    /*InputLayer inp(2);
    FeedForwardLayer hidden(&inp, 2);
    FeedForwardLayer output(&hidden, 2);

    hidden.GetWeights()->SetValue(0, 0, 0.15);
    hidden.GetWeights()->SetValue(1, 0, 0.2);
    hidden.GetWeights()->SetValue(0, 1, 0.25);
    hidden.GetWeights()->SetValue(1, 1, 0.3);

    output.GetWeights()->SetValue(0, 0, 0.4);
    output.GetWeights()->SetValue(1, 0, 0.45);
    output.GetWeights()->SetValue(0, 1, 0.5);
    output.GetWeights()->SetValue(1, 1, 0.55);

    hidden.GetBias()->SetValue(0, 0.35);
    hidden.GetBias()->SetValue(1, 0.35);

    output.GetBias()->SetValue(0, 0.6);
    output.GetBias()->SetValue(1, 0.6);

    hidden.SetActivationFunction(new Sigmoid());
    output.SetActivationFunction(new Sigmoid());

    Matrix input(1, 2);
    input[0] = 0.05;
    input[1] = 0.1;

    inp.SetInput(&input);
    Matrix* outval = output.ComputeAndGetOutput();
    std::cout << "Before training:" << std::endl;
    MatrixMath::PrintMatrix(outval);

    //start training

    Matrix expected(1, 2);
    expected[0] = 0.01;
    expected[1] = 0.99;

    GradientDescent trainer(LossFunctions::MSE, LossFunctions::MSE_Derivate, &output, 0.5);
    for (size_t i = 0; i < 500; i++)
    {
        trainer.Train(&input, &expected);
    }
    std::cout << "After training:" << std::endl;
    output.Compute();
    MatrixMath::PrintMatrix(outval);*/

    InputLayer inp(2);
    RecurrentLayer rec(std::make_shared<InputLayer>(inp), 5);

    Matrix input(1, 2);
    input[0] = 0.05;
    input[1] = 0.1;

    rec.SetTrainingMode(true);

    inp.SetInput(std::make_shared<Matrix>(input));
    rec.Compute();


    std::cout << "Hello World!\n";
}
