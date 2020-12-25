#pragma once
#include "Layer.h"
#include "Matrix.h"
#include "ActivationFunctions.hpp"
#include <vector>
#include <deque>

class LSTM :
    public Layer
{
public:
    LSTM(Layer* inputLayer, unsigned int cellStateSize, unsigned int timeSteps = 3);
    virtual ~LSTM();
    virtual Layer* Clone();

    virtual void Compute();
    virtual Matrix* GetOutput();
    virtual Matrix* ComputeAndGetOutput();
    
    virtual void GetBackwardPass(Matrix* error, bool recursive = false);

    virtual void Train(Optimizer* optimizer);
    virtual void SetTrainingMode(bool mode);

    Matrix* GetWeight(unsigned char weight);
    Matrix* GetRecursiveWeight(unsigned char weight);
    Matrix* GetBias(unsigned char weight);

private:
    std::vector<Matrix*> InputWeights;
    std::vector<Matrix*> RecursiveWeights;
    std::vector<Matrix*> Biases;
    std::vector<Matrix*> InputWeightOutputs;
    std::vector<Matrix*> RecursiveWeightOuputs;

    std::vector<Matrix*> InputWeightErrors;
    std::vector<Matrix*> RecursiveWeightErrors;
    std::vector<Matrix*> BiasErrors;

    std::deque<std::vector<Matrix*>> savedStates;
    std::deque<Matrix*> errors;

    Matrix* CellState;
    Matrix* RecurrentState;
    Matrix* InnerState;

    Matrix* cellTanh;
    Matrix* DeltaOut;

    unsigned int CellStateSize;
    unsigned int TimeSteps;

    ActivationFunction* Tanh;
    ActivationFunction* Sig;

    void UpdateWeightErrors(Matrix* gateIError, Matrix* gateRError, Matrix* inputTranspose, Matrix* dGate, Matrix* outputTranspose, int weight);
};

