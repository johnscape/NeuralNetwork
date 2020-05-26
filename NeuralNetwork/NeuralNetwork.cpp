// NeuralNetwork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "Matrix.h"
#include "InputLayer.h"
#include "FeedForwardLayer.h"
#include "ActivationFunctions.hpp"
#include "Constants.h"

int main()
{
    Matrix* t = new Matrix(5, 5);
    delete t;

    InputLayer inp(2);
    FeedForwardLayer hidden(&inp, 3);
    FeedForwardLayer output(&hidden, 1);

    Matrix a(2, 1);

    a[0] = 1;
    a[1] = 1;

    hidden.SetActivationFunction(new Sigmoid());
    output.SetActivationFunction(new Sigmoid());

    inp.SetInput(&a);
    
    hidden.GetBias()->SetValue(0, 0);
    hidden.GetBias()->SetValue(1, 0);
    hidden.GetBias()->SetValue(2, 0);

    output.GetBias()->SetValue(0, 0);

    for (unsigned int i = 0; i < hidden.GetWeights()->GetRowCount(); i++)
        for (unsigned int j = 0; j < hidden.GetWeights()->GetColumnCount(); j++)
            hidden.GetWeights()->SetValue(i, j, 1);

    for (unsigned int i = 0; i < output.GetWeights()->GetRowCount(); i++)
        for (unsigned int j = 0; j < output.GetWeights()->GetColumnCount(); j++)
            output.GetWeights()->SetValue(i, j, 1);

    Matrix* o = output.GetOutput();


    std::cout << "Hello World!\n";

    o = nullptr;
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
