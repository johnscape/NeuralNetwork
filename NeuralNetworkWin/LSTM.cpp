#include "LSTM.h"
#include "Optimizer.h"
#include "Constants.h"

#if USE_GPU
#include "MatrixGPUMath.cuh"
#endif // USE_GPU


LSTM::LSTM(Layer* inputLayer, unsigned int cellStateSize, unsigned int timeSteps) : 
    Layer(inputLayer), CellStateSize(cellStateSize), TimeSteps(timeSteps), cellTanh(1, cellStateSize),
    DeltaOut(1, cellStateSize), CellState(1, cellStateSize), InnerState(1, cellStateSize)
{
    for (unsigned char i = 0; i < 4; i++)
    {
        InputWeights.push_back(Matrix(LayerInput->GetOutput().GetVectorSize(), CellStateSize));
        RecursiveWeights.push_back(Matrix(CellStateSize, CellStateSize));
        MatrixMath::FillWithRandom(InputWeights[InputWeights.size() - 1]);
        MatrixMath::FillWithRandom(RecursiveWeights[RecursiveWeights.size() - 1]);
        Biases.push_back(Matrix(1, CellStateSize));
        InputWeightOutputs.push_back(Matrix(1, CellStateSize));
        RecursiveWeightOuputs.push_back(Matrix(1, CellStateSize));

        InputWeightErrors.push_back(Matrix(LayerInput->GetOutput().GetVectorSize(), CellStateSize));
        RecursiveWeightErrors.push_back(Matrix(CellStateSize, CellStateSize));
        BiasErrors.push_back(Matrix(1, CellStateSize));
    }
    

    Output.Reset(1, CellStateSize);

    Tanh = &TanhFunction::GetInstance();
    Sig = &Sigmoid::GetInstance();

    LayerError.Reset(1, LayerInput->GetOutput().GetVectorSize());
}

LSTM::~LSTM()
{
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
#if USE_GPU
        GPUMath::FillWith(InputWeightOutputs[i], 0);
        GPUMath::FillWith(RecursiveWeightOuputs[i], 0);
#endif // USE_GPU
    }

    LayerInput->Compute();
    Matrix input = LayerInput->GetOutput();
    std::vector<Matrix> currentStates;
    
    //forget gate
    MatrixMath::Multiply(input, InputWeights[0], InputWeightOutputs[0]);
    MatrixMath::Multiply(InnerState, RecursiveWeights[0], RecursiveWeightOuputs[0]);
    MatrixMath::AddIn(InputWeightOutputs[0], RecursiveWeightOuputs[0]);
    MatrixMath::AddIn(InputWeightOutputs[0], Biases[0]);
    Sig->CalculateInto(InputWeightOutputs[0], InputWeightOutputs[0]);
    MatrixMath::ElementviseMultiply(CellState, InputWeightOutputs[0]);
    if (TrainingMode)
        currentStates.push_back(Matrix(InputWeightOutputs[0]));

    //input gate
    MatrixMath::Multiply(input, InputWeights[1], InputWeightOutputs[1]);
    MatrixMath::Multiply(InnerState, RecursiveWeights[1], RecursiveWeightOuputs[1]);
    MatrixMath::AddIn(InputWeightOutputs[1], RecursiveWeightOuputs[1]);
    MatrixMath::AddIn(InputWeightOutputs[1], Biases[1]);
    Sig->CalculateInto(InputWeightOutputs[1], InputWeightOutputs[1]);
    if (TrainingMode)
        currentStates.push_back(Matrix(InputWeightOutputs[1]));

    //update gate
    MatrixMath::Multiply(input, InputWeights[2], InputWeightOutputs[2]);
    MatrixMath::Multiply(InnerState, RecursiveWeights[2], RecursiveWeightOuputs[2]);
    MatrixMath::AddIn(InputWeightOutputs[2], RecursiveWeightOuputs[2]);
    MatrixMath::AddIn(InputWeightOutputs[2], Biases[2]);
    Tanh->CalculateInto(InputWeightOutputs[2], InputWeightOutputs[2]);
    if (TrainingMode)
        currentStates.push_back(Matrix(InputWeightOutputs[2]));
    MatrixMath::ElementviseMultiply(InputWeightOutputs[2], InputWeightOutputs[1]);
    MatrixMath::AddIn(CellState, InputWeightOutputs[2]);

    //output gate
    MatrixMath::Multiply(input, InputWeights[3], InputWeightOutputs[3]);
    MatrixMath::Multiply(InnerState, RecursiveWeights[3], RecursiveWeightOuputs[3]);
    MatrixMath::AddIn(InputWeightOutputs[3], RecursiveWeightOuputs[3]);
    MatrixMath::AddIn(InputWeightOutputs[3], Biases[3]);
    Sig->CalculateInto(InputWeightOutputs[3], InputWeightOutputs[3]);
    if (TrainingMode)
        currentStates.push_back(Matrix(InputWeightOutputs[3]));

    //output
    Tanh->CalculateInto(CellState, cellTanh);
    MatrixMath::ElementviseMultiply(InputWeightOutputs[3], cellTanh);
    MatrixMath::Copy(InputWeightOutputs[3], InnerState);
    MatrixMath::Copy(InnerState, Output);
    if (TrainingMode)
    {
        currentStates.push_back(Matrix(CellState));
        currentStates.push_back(Matrix(input));
        currentStates.push_back(Matrix(Output));
        savedStates.push_back(currentStates);
        if (savedStates.size() >= TimeSteps)
        {
            savedStates.pop_front();
        }
    }
}

Matrix& LSTM::GetOutput()
{
    return Output;
}

Matrix& LSTM::ComputeAndGetOutput()
{
    Compute();
    return Output;
}

void LSTM::GetBackwardPass(const Matrix& error, bool recursive)
{
#if USE_GPU
    error->CopyToGPU();
#endif // USE_GPU

    Matrix gateIError(LayerInput->GetOutput().GetVectorSize(), CellStateSize);
    Matrix gateRError(CellStateSize, CellStateSize);

    Matrix dGate(1, CellStateSize);

    Matrix deltaOut(1, CellStateSize);
    Matrix dOut(1, CellStateSize);

    Matrix dState(1, CellStateSize);
    Matrix dStateLast(1, CellStateSize); //dState in the last iteration, t+1

    Matrix tanhState(1, CellStateSize);
    Matrix tempState(1, CellStateSize);
    Matrix ones(1, CellStateSize);

    Matrix inputTranspose(LayerInput->GetOutput().GetVectorSize(), 1);
    Matrix outputTranspose(CellStateSize, 1);

    Matrix inputErrorSum(LayerInput->GetOutput().GetVectorSize(), 1);

    errors.push_back(Matrix(error));
    if (errors.size() >= TimeSteps)
        errors.pop_front();

    for (signed int time = TimeSteps - 1; time >= 0; time--)
    {
        if (time >= savedStates.size())
            continue;

        //setup transposes vals
        inputTranspose.ReloadFromOther(savedStates[time][5]);
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
            //Matrix fGateiDelta = new Matrix(*savedDStates[savedDStates.size() - 1]);
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

#if USE_GPU
    LayerError->CopyFromGPU();
#endif // USE_GPU
}

void LSTM::UpdateWeightErrors(Matrix& gateIError, Matrix& gateRError, Matrix& inputTranspose, Matrix& dGate, Matrix& outputTranspose, int weight)
{
    MatrixMath::FillWith(gateIError, 0);
    MatrixMath::FillWith(gateRError, 0);

    MatrixMath::Multiply(inputTranspose, dGate, gateIError);
    MatrixMath::Multiply(outputTranspose, dGate, gateRError);

    MatrixMath::AddIn(InputWeightErrors[weight], gateIError);
    MatrixMath::AddIn(RecursiveWeightErrors[weight], gateRError);

    MatrixMath::AddIn(BiasErrors[weight], dGate);

#if USE_GPU
    InputWeightErrors[weight]->CopyFromGPU();
    RecursiveWeightErrors[weight]->CopyFromGPU();
    BiasErrors[weight]->CopyFromGPU();
#endif // USE_GPU

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

void LSTM::SetTrainingMode(bool mode, bool recursive)
{
    TrainingMode = mode;
    MatrixMath::FillWith(InnerState, 0);
    MatrixMath::FillWith(CellState, 0);
    if (recursive && LayerInput)
        LayerInput->SetTrainingMode(mode, recursive);
}

Matrix& LSTM::GetWeight(unsigned char weight)
{
    return InputWeights[weight];
}

Matrix& LSTM::GetRecursiveWeight(unsigned char weight)
{
    return RecursiveWeights[weight];
}

Matrix& LSTM::GetBias(unsigned char weight)
{
    return Biases[weight];
}

void LSTM::LoadFromJSON(const char* data, bool isFile)
{
    rapidjson::Document document;
    if (!isFile)
        document.Parse(data);
    else
    {
        std::ifstream r(data);
        rapidjson::IStreamWrapper isw(r);
        document.ParseStream(isw);
    }
    rapidjson::Value val;

    InputWeights.clear();
    Biases.clear();
    InputWeightOutputs.clear();
    InputWeightErrors.clear();
    BiasErrors.clear();
    RecursiveWeights.clear();
    RecursiveWeightErrors.clear();
    RecursiveWeightOuputs.clear();

    unsigned int InputSize = 1;
    val = document["layer"]["size"];
    CellStateSize = val.GetUint();
    val = document["layer"]["inputSize"];

    unsigned int inputSize = val.GetUint();
    if (LayerInput)
        inputSize = LayerInput->GetOutput().GetVectorSize();
    for (unsigned char i = 0; i < 4; i++)
    {
        InputWeights.push_back(Matrix(inputSize, CellStateSize));
        RecursiveWeights.push_back(Matrix(CellStateSize, CellStateSize));
        Biases.push_back(Matrix(1, CellStateSize));
        InputWeightOutputs.push_back(Matrix(1, CellStateSize));
        RecursiveWeightOuputs.push_back(Matrix(1, CellStateSize));

        InputWeightErrors.push_back(Matrix(inputSize, CellStateSize));
        RecursiveWeightErrors.push_back(Matrix(CellStateSize, CellStateSize));
        BiasErrors.push_back(Matrix(1, CellStateSize));
    }

    Output.Reset(1, CellStateSize);
    cellTanh.Reset(1, CellStateSize);
    DeltaOut.Reset(1, CellStateSize);

    CellState.Reset(1, CellStateSize);
    InnerState.Reset(1, CellStateSize);

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);

    std::string varName;

    for (unsigned char i = 0; i < 4; i++)
    {

        varName = "inputWeight" + std::to_string((i + 1));
        document["layer"][varName.c_str()].Accept(writer);
        InputWeights[i].LoadFromJSON(buffer.GetString());

        buffer.Clear();
        writer.Reset(buffer);

        varName = "recursiveWeight" + std::to_string((i + 1));
        document["layer"][varName.c_str()].Accept(writer);
        RecursiveWeights[i].LoadFromJSON(buffer.GetString());

        buffer.Clear();
        writer.Reset(buffer);

        varName = "bias" + std::to_string((i + 1));
        document["layer"][varName.c_str()].Accept(writer);
        Biases[i].LoadFromJSON(buffer.GetString());

        buffer.Clear();
        writer.Reset(buffer);
    }
}

std::string LSTM::SaveToJSON(const char* fileName)
{
    rapidjson::Document doc;
    doc.SetObject();

    rapidjson::Value layerSize, id, type, inputSize;
    layerSize.SetUint(CellStateSize);
    id.SetUint(Id);
    type.SetUint(3);
    if (LayerInput)
        inputSize.SetUint(LayerInput->GetOutput().GetVectorSize());
    else
        inputSize.SetUint(1);

    rapidjson::Document inp1, inp2, inp3, inp4, rec1, rec2, rec3, rec4;
    rapidjson::Document b1, b2, b3, b4;


    inp1.Parse(InputWeights[0].SaveToJSON().c_str());
    inp2.Parse(InputWeights[1].SaveToJSON().c_str());
    inp3.Parse(InputWeights[2].SaveToJSON().c_str());
    inp4.Parse(InputWeights[3].SaveToJSON().c_str());

    rec1.Parse(RecursiveWeights[0].SaveToJSON().c_str());
    rec2.Parse(RecursiveWeights[1].SaveToJSON().c_str());
    rec3.Parse(RecursiveWeights[2].SaveToJSON().c_str());
    rec4.Parse(RecursiveWeights[3].SaveToJSON().c_str());

    b1.Parse(Biases[0].SaveToJSON().c_str());
    b2.Parse(Biases[1].SaveToJSON().c_str());
    b3.Parse(Biases[2].SaveToJSON().c_str());
    b4.Parse(Biases[3].SaveToJSON().c_str());

    rapidjson::Value root(rapidjson::kObjectType);
    root.AddMember("id", id, doc.GetAllocator());
    root.AddMember("type", type, doc.GetAllocator());
    root.AddMember("size", layerSize, doc.GetAllocator());
    root.AddMember("inputSize", inputSize, doc.GetAllocator());
    root.AddMember("inputWeight1", inp1, doc.GetAllocator());
    root.AddMember("inputWeight2", inp2, doc.GetAllocator());
    root.AddMember("inputWeight3", inp3, doc.GetAllocator());
    root.AddMember("inputWeight4", inp4, doc.GetAllocator());
    root.AddMember("recursiveWeight1", rec1, doc.GetAllocator());
    root.AddMember("recursiveWeight2", rec2, doc.GetAllocator());
    root.AddMember("recursiveWeight3", rec3, doc.GetAllocator());
    root.AddMember("recursiveWeight4", rec4, doc.GetAllocator());
    root.AddMember("bias1", b1, doc.GetAllocator());
    root.AddMember("bias2", b2, doc.GetAllocator());
    root.AddMember("bias3", b3, doc.GetAllocator());
    root.AddMember("bias4", b4, doc.GetAllocator());

    doc.AddMember("layer", root, doc.GetAllocator());

    if (fileName)
    {
        std::ofstream w(fileName);
        rapidjson::OStreamWrapper osw(w);
        rapidjson::Writer<rapidjson::OStreamWrapper> writer(osw);
        doc.Accept(writer);
        w.close();
    }

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);

    return std::string(buffer.GetString());
}
