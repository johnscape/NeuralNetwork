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

    MatrixMath::FillWithRandom(hidden.GetWeights());
    MatrixMath::FillWithRandom(hidden.GetBias());
    MatrixMath::FillWithRandom(output.GetWeights());
    MatrixMath::FillWithRandom(output.GetBias());

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
    expected[1] = 0.99;*/

    //GradientDescent trainer(LossFunctions::MSE, LossFunctions::MSE_Derivate, &output, 0.5);
    /*for (size_t i = 0; i < 500; i++)
    {
        trainer.Train(&input, &expected);
    }*/
    /*trainer.TrainFor(&input, &expected, 500);
    std::cout << "After training:" << std::endl;
    output.Compute();
    MatrixMath::PrintMatrix(outval);*/

    InputLayer inp(2);
    RecurrentLayer rec(&inp, 4);

    MatrixMath::FillWithRandom(rec.GetWeights());
    MatrixMath::FillWithRandom(rec.GetBias());
    MatrixMath::FillWithRandom(rec.GetRecurrentWeights());

    Matrix input1(1, 2);
    input1[0] = 0;
    input1[1] = 1;

    rec.SetActivationFunction(new Sigmoid());

    inp.SetInput(&input1);
    Matrix* out1 = rec.ComputeAndGetOutput();
    std::cout << "Before training:" << std::endl;
    MatrixMath::PrintMatrix(out1);

    Matrix trainingSet(32 * 50, 2);
    Matrix trainingOutput(32 * 50, 4);

    trainingSet[0] = 0;
    trainingSet[1] = 1;

    trainingOutput[0] = 0;
    trainingOutput[1] = 1;
    trainingOutput[2] = 0;
    trainingOutput[3] = 0;

    srand(time(0));

    for (size_t i = 1; i < 32 * 50; i++)
    {
        trainingSet.SetValue(i, 0, rand() % 2);
        trainingSet.SetValue(i, 1, rand() % 2);

        trainingOutput.SetValue(i, 0, trainingSet.GetValue(i, 0));
        trainingOutput.SetValue(i, 1, trainingSet.GetValue(i, 1));
        trainingOutput.SetValue(i, 2, trainingSet.GetValue(i - 1, 0));
        trainingOutput.SetValue(i, 3, trainingSet.GetValue(i - 2, 1));
    }

    GradientDescent trainer(LossFunctions::MSE, LossFunctions::MSE_Derivate, &rec, 0.5);
    trainer.TrainFor(&trainingSet, &trainingOutput, 100);
    std::cout << "After training: " << std::endl;
    inp.SetInput(&input1);
    out1 = rec.ComputeAndGetOutput();
    MatrixMath::PrintMatrix(out1);
    //trainer.Train(input, expected);
    //trainer.Reset();
}
