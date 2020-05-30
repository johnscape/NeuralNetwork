// NeuralNetwork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "Matrix.h"
#include "InputLayer.h"
#include "FeedForwardLayer.h"
#include "ActivationFunctions.hpp"
#include "Constants.h"

#include "GradientDescent.h"
#include "LossFunctions.hpp"

//TODO: Freeze layers

int main()
{
    InputLayer inp(2);
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
    Matrix* outval = output.GetOutput();

    //start training

    Matrix expected(1, 2);
    expected[0] = 0.01;
    expected[1] = 0.99;

    GradientDescent trainer(LossFunctions::MSE, LossFunctions::MSE_Derivate, &output, 0.5);
    trainer.Train(&input, &expected);

    std::cout << "Hello World!\n";
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
