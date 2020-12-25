#include "LSTM.h"
#include "Optimizer.h"

LSTM::LSTM(Layer* inputLayer, unsigned int cellStateSize, unsigned int timeSteps) : Layer(inputLayer), CellStateSize(cellStateSize), TimeSteps(timeSteps)
{
    for (unsigned char i = 0; i < 4; i++)
    {
        InputWeights.push_back(new Matrix(LayerInput->GetOutput()->GetVectorSize(), CellStateSize));
        RecursiveWeights.push_back(new Matrix(CellStateSize, CellStateSize));
        MatrixMath::FillWithRandom(InputWeights[InputWeights.size() - 1]);
        MatrixMath::FillWithRandom(RecursiveWeights[RecursiveWeights.size() - 1]);
        Biases.push_back(new Matrix(1, CellStateSize));
        InputWeightOutputs.push_back(new Matrix(1, CellStateSize));
        RecursiveWeightOuputs.push_back(new Matrix(1, CellStateSize));

        InputWeightErrors.push_back(new Matrix(LayerInput->GetOutput()->GetVectorSize(), CellStateSize));
        RecursiveWeightErrors.push_back(new Matrix(CellStateSize, CellStateSize));
        BiasErrors.push_back(new Matrix(1, CellStateSize));
    }
    

    Output = new Matrix(1, CellStateSize);
    cellTanh = new Matrix(1, CellStateSize);
    DeltaOut = new Matrix(1, CellStateSize);

    CellState = new Matrix(1, CellStateSize);
    RecurrentState = new Matrix(1, CellStateSize);
    InnerState = new Matrix(1, CellStateSize);

    Tanh = new TanhFunction();
    Sig = new Sigmoid();

    LayerError = new Matrix(1, LayerInput->GetOutput()->GetVectorSize());
}

LSTM::~LSTM()
{
    delete CellState;
    delete RecurrentState;
    delete InnerState;
    delete cellTanh;
    delete DeltaOut;

    delete Tanh;
    delete Sig;
    for (unsigned char i = 0; i < 4; i++)
    {
        delete InputWeights[i];
        delete Biases[i];
        delete InputWeightOutputs[i];
        delete InputWeightErrors[i];
        delete BiasErrors[i];
        delete RecursiveWeights[i];
        delete RecursiveWeightErrors[i];
        delete RecursiveWeightOuputs[i];
    }
}

Layer* LSTM::Clone()
{
    LSTM* r = new LSTM(LayerInput, CellStateSize, TimeSteps);
    for (unsigned char i = 0; i < 4; i++)
    {
        MatrixMath::Copy(InputWeights[i], r->GetWeight(i));
        MatrixMath::Copy(RecursiveWeights[i], r->GetRecursiveWeight(i));
        MatrixMath::Copy(Biases[i], r->GetBias(i));
    }
    return r;
}

void LSTM::Compute()
{
    for (unsigned char i = 0; i < 4; i++)
    {
        MatrixMath::FillWith(InputWeightOutputs[i], 0);
        MatrixMath::FillWith(RecursiveWeightOuputs[i], 0);
    }

    LayerInput->Compute();
    Matrix* input = LayerInput->GetOutput();
    std::vector<Matrix*> currentStates;

    //forget gate
    MatrixMath::Multiply(input, InputWeights[0], InputWeightOutputs[0]);
    MatrixMath::Multiply(InnerState, RecursiveWeights[0], RecursiveWeightOuputs[0]);
    MatrixMath::AddIn(InputWeightOutputs[0], RecursiveWeightOuputs[0]);
    MatrixMath::AddIn(InputWeightOutputs[0], Biases[0]);
    Sig->CalculateInto(InputWeightOutputs[0], InputWeightOutputs[0]);
    MatrixMath::ElementviseMultiply(CellState, InputWeightOutputs[0]);
    if (TrainingMode)
        currentStates.push_back(new Matrix(*InputWeightOutputs[0]));

    //input gate
    MatrixMath::Multiply(input, InputWeights[1], InputWeightOutputs[1]);
    MatrixMath::Multiply(InnerState, RecursiveWeights[1], RecursiveWeightOuputs[1]);
    MatrixMath::AddIn(InputWeightOutputs[1], RecursiveWeightOuputs[1]);
    MatrixMath::AddIn(InputWeightOutputs[1], Biases[1]);
    Sig->CalculateInto(InputWeightOutputs[1], InputWeightOutputs[1]);
    if (TrainingMode)
        currentStates.push_back(new Matrix(*InputWeightOutputs[1]));

    //update gate
    MatrixMath::Multiply(input, InputWeights[2], InputWeightOutputs[2]);
    MatrixMath::Multiply(InnerState, RecursiveWeights[2], RecursiveWeightOuputs[2]);
    MatrixMath::AddIn(InputWeightOutputs[2], RecursiveWeightOuputs[2]);
    MatrixMath::AddIn(InputWeightOutputs[2], Biases[2]);
    Tanh->CalculateInto(InputWeightOutputs[2], InputWeightOutputs[2]);
    if (TrainingMode)
        currentStates.push_back(new Matrix(*InputWeightOutputs[2]));
    MatrixMath::ElementviseMultiply(InputWeightOutputs[2], InputWeightOutputs[1]);
    MatrixMath::AddIn(CellState, InputWeightOutputs[2]);

    //output gate
    MatrixMath::Multiply(input, InputWeights[3], InputWeightOutputs[3]);
    MatrixMath::Multiply(InnerState, RecursiveWeights[3], RecursiveWeightOuputs[3]);
    MatrixMath::AddIn(InputWeightOutputs[3], RecursiveWeightOuputs[3]);
    MatrixMath::AddIn(InputWeightOutputs[3], Biases[3]);
    Sig->CalculateInto(InputWeightOutputs[3], InputWeightOutputs[3]);
    if (TrainingMode)
        currentStates.push_back(new Matrix(*InputWeightOutputs[3]));

    //output
    Tanh->CalculateInto(CellState, cellTanh);
    MatrixMath::ElementviseMultiply(InputWeightOutputs[3], cellTanh);
    MatrixMath::Copy(InputWeightOutputs[3], InnerState);
    MatrixMath::Copy(InnerState, Output);
    if (TrainingMode)
    {
        currentStates.push_back(new Matrix(*CellState));
        currentStates.push_back(new Matrix(*input));
        currentStates.push_back(new Matrix(*Output));
        savedStates.push_back(currentStates);
        if (savedStates.size() >= TimeSteps)
        {
            for (size_t i = 0; i < savedStates[0].size(); i++)
                delete savedStates[0][i];
            savedStates.pop_front();
        }
    }
}

Matrix* LSTM::GetOutput()
{
    return Output;
}

Matrix* LSTM::ComputeAndGetOutput()
{
    Compute();
    return Output;
}

void LSTM::GetBackwardPass(Matrix* error, bool recursive)
{
    Matrix* gateIError = new Matrix(LayerInput->GetOutput()->GetVectorSize(), CellStateSize);
    Matrix* gateRError = new Matrix(CellStateSize, CellStateSize);

    Matrix* dGate = new Matrix(1, CellStateSize);

    Matrix* deltaOut = new Matrix(1, CellStateSize);
    Matrix* dOut = new Matrix(1, CellStateSize);

    Matrix* dState = new Matrix(1, CellStateSize);
    Matrix* dStateLast = new Matrix(1, CellStateSize); //dState in the last iteration, t+1

    Matrix* tanhState = new Matrix(1, CellStateSize);
    Matrix* tempState = new Matrix(1, CellStateSize);
    Matrix* ones = new Matrix(1, CellStateSize);

    Matrix* inputTranspose = new Matrix(LayerInput->GetOutput()->GetVectorSize(), 1);
    Matrix* outputTranspose = new Matrix(CellStateSize, 1);

    Matrix* inputErrorSum = new Matrix(LayerInput->GetOutput()->GetVectorSize(), 1);

    errors.push_back(new Matrix(*error));
    if (errors.size() >= TimeSteps)
    {
        delete errors[0];
        errors.pop_front();
    }

    for (signed int time = TimeSteps - 1; time >= 0; time--)
    {
        if (time >= savedStates.size())
            continue;

        //setup transposes vals
        inputTranspose->ReloadFromOther(savedStates[time][5]);
        MatrixMath::Copy(savedStates[time][5], inputTranspose);
        if (time - 1 >= 0)
            MatrixMath::Copy(savedStates[time - 1][6], outputTranspose);
        else
            MatrixMath::FillWith(outputTranspose, 0);

        //calculate delta Out
        MatrixMath::FillWith(dOut, 0);
        MatrixMath::AddIn(dOut, deltaOut);
        MatrixMath::AddIn(dOut, errors[time]);

        //calculate delta State
        MatrixMath::Copy(dState, dStateLast);
        MatrixMath::FillWith(dState, 0);
        MatrixMath::AddIn(dState, dOut);
        MatrixMath::ElementviseMultiply(dState, savedStates[time][3]);

        MatrixMath::Copy(savedStates[time][4], tanhState);
        Tanh->CalculateInto(tanhState, tanhState);
        Tanh->CalculateDerivateInto(tanhState, tanhState);
        MatrixMath::ElementviseMultiply(dState, tanhState);

        MatrixMath::FillWith(deltaOut, 0);

        if (time < savedStates.size() - 1)
        {
            MatrixMath::Copy(dStateLast, tempState);
            MatrixMath::ElementviseMultiply(tempState, savedStates[time + 1][0]);
            MatrixMath::AddIn(dState, tempState);
        }

        //calc forget gate error
        if (time > 0)
        {
            //Matrix* fGateiDelta = new Matrix(*savedDStates[savedDStates.size() - 1]);
            MatrixMath::Copy(dState, dGate);
            MatrixMath::ElementviseMultiply(dGate, savedStates[time - 1][4]);
            MatrixMath::ElementviseMultiply(dGate, savedStates[time][0]);
            MatrixMath::FillWith(ones, 1);
            MatrixMath::SubstractIn(ones, savedStates[time][0]);
            MatrixMath::ElementviseMultiply(dGate, ones);

            UpdateWeightErrors(gateIError, gateRError, inputTranspose, dGate, outputTranspose, 0);
            MatrixMath::Multiply(dGate, RecursiveWeights[0], deltaOut);
            MatrixMath::Transpose(dGate);
            MatrixMath::Multiply(InputWeights[0], dGate, inputErrorSum);
            MatrixMath::Transpose(dGate);
        }

        //calc input gate error
        MatrixMath::Copy(dState, dGate);
        MatrixMath::ElementviseMultiply(dGate, savedStates[time][2]);
        MatrixMath::ElementviseMultiply(dGate, savedStates[time][1]);
        MatrixMath::FillWith(ones, 1);
        MatrixMath::SubstractIn(ones, savedStates[time][1]);
        MatrixMath::ElementviseMultiply(dGate, ones);

        UpdateWeightErrors(gateIError, gateRError, inputTranspose, dGate, outputTranspose, 1);
        MatrixMath::Multiply(dGate, RecursiveWeights[1], deltaOut);
        MatrixMath::Transpose(dGate);
        MatrixMath::Multiply(InputWeights[1], dGate, inputErrorSum);
        MatrixMath::Transpose(dGate);

        //calc update gate error
        MatrixMath::Copy(dState, dGate);
        MatrixMath::ElementviseMultiply(dGate, savedStates[time][1]);
        MatrixMath::Copy(savedStates[time][2], ones);
        Tanh->CalculateDerivateInto(ones, ones);
        MatrixMath::ElementviseMultiply(dGate, ones);

        UpdateWeightErrors(gateIError, gateRError, inputTranspose, dGate, outputTranspose, 2);
        MatrixMath::Multiply(dGate, RecursiveWeights[2], deltaOut);
        MatrixMath::Transpose(dGate);
        MatrixMath::Multiply(InputWeights[2], dGate, inputErrorSum);
        MatrixMath::Transpose(dGate);

        //calc output gate error
        MatrixMath::Copy(dOut, dGate);
        MatrixMath::Copy(savedStates[time][4], ones);
        Tanh->CalculateInto(ones, ones);
        MatrixMath::ElementviseMultiply(dGate, ones);
        MatrixMath::ElementviseMultiply(dGate, savedStates[time][3]);
        MatrixMath::FillWith(ones, 1);
        MatrixMath::SubstractIn(ones, savedStates[time][3]);
        MatrixMath::ElementviseMultiply(dGate, ones);

        UpdateWeightErrors(gateIError, gateRError, inputTranspose, dGate, outputTranspose, 3);
        MatrixMath::Multiply(dGate, RecursiveWeights[3], deltaOut);
        MatrixMath::Transpose(dGate);
        MatrixMath::Multiply(InputWeights[3], dGate, inputErrorSum);
        MatrixMath::Transpose(dGate);
    }

    MatrixMath::Transpose(inputErrorSum);
    MatrixMath::AddIn(LayerError, inputErrorSum);

    delete gateIError;
    delete gateRError;
    delete dGate;
    delete deltaOut;
    delete dOut;
    delete dState;
    delete dStateLast;
    delete tanhState;
    delete tempState;
    delete ones;
    delete inputTranspose;
    delete outputTranspose;
    delete inputErrorSum;
}

void LSTM::UpdateWeightErrors(Matrix* gateIError, Matrix* gateRError, Matrix* inputTranspose, Matrix* dGate, Matrix* outputTranspose, int weight)
{
    MatrixMath::FillWith(gateIError, 0);
    MatrixMath::FillWith(gateRError, 0);

    MatrixMath::Multiply(inputTranspose, dGate, gateIError);
    MatrixMath::Multiply(outputTranspose, dGate, gateRError);

    MatrixMath::AddIn(InputWeightErrors[weight], gateIError);
    MatrixMath::AddIn(RecursiveWeightErrors[weight], gateRError);

    MatrixMath::AddIn(BiasErrors[weight], dGate);
}

void LSTM::Train(Optimizer* optimizer)
{
    for (unsigned char i = 0; i < 4; i++)
    {
        optimizer->ModifyWeights(InputWeights[i], InputWeightErrors[i]);
        optimizer->ModifyWeights(RecursiveWeights[i], RecursiveWeightErrors[i]);
        optimizer->ModifyWeights(Biases[i], BiasErrors[i]);

        MatrixMath::FillWith(InputWeightErrors[i], 0);
        MatrixMath::FillWith(RecursiveWeightErrors[i], 0);
        MatrixMath::FillWith(BiasErrors[i], 0);
    }
}

void LSTM::SetTrainingMode(bool mode)
{
    TrainingMode = mode;
}

Matrix* LSTM::GetWeight(unsigned char weight)
{
    if (weight > 3)
        return nullptr;
    return InputWeights[weight];
}

Matrix* LSTM::GetRecursiveWeight(unsigned char weight)
{
    if (weight > 3)
        return nullptr;
    return RecursiveWeights[weight];
}

Matrix* LSTM::GetBias(unsigned char weight)
{
    if (weight > 3)
        return nullptr;
    return Biases[weight];
}
