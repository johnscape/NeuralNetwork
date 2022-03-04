#include "NeuralNetwork/LSTM.h"
#include "NeuralNetwork/Optimizer.h"
#include "NeuralNetwork/Constants.h"

#if USE_GPU
#include "MatrixGPUMath.cuh"
#endif // USE_GPU


LSTM::LSTM(Layer* inputLayer, unsigned int cellStateSize, unsigned int timeSteps) : 
    Layer(inputLayer), CellStateSize(cellStateSize), TimeSteps(timeSteps), CellState(1, cellStateSize),
	InnerState(1, inputLayer->GetOutput().GetShapeAt(0) + cellStateSize)
{
	for (unsigned char i = 0; i < 4; i++)
	{
		Matrix w(InnerState.GetColumnCount(), cellStateSize);
		Matrix b(1, cellStateSize);
		Weights.push_back(w);
		Biases.push_back(b);
	}
    Output = Tensor({1, CellStateSize}, nullptr);

    Tanh = &TanhFunction::GetInstance();
    Sig = &Sigmoid::GetInstance();

    LayerError = Tensor({1, LayerInput->GetOutput().GetShapeAt(0)}, nullptr);
}

LSTM::~LSTM()
{
}

Layer* LSTM::Clone()
{
    LSTM* r = new LSTM(LayerInput, CellStateSize, TimeSteps);
	std::list<Matrix>::iterator w, b;
    for (unsigned char i = 0; i < 4; i++)
    {
        r->GetWeight(i).Copy(*w);
        r->GetBias(i).Copy(*b);
		w++; b++;
    }
    return r;
}

void LSTM::Compute()
{
	Matrix input = LayerInput->ComputeAndGetOutput().ToMatrixByRows();
	Matrix tempOutput(input.GetRowCount(), CellStateSize);
	Matrix gate1, gate2, gate3, gate4;
	Matrix cellTanh(1, CellStateSize);
	for (size_t r = 0; r < input.GetRowCount(); ++r)
	{
		Matrix row = input.GetRowMatrix(r);
		InnerState = Matrix::Concat(InnerState, row, 1);
		std::list<Matrix>::iterator weight, bias;
		weight = Weights.begin();
		bias = Biases.begin();
		//Gate 1
		gate1 = InnerState * *weight;
		gate1 += *bias;
		gate1 = Sig->CalculateMatrix(gate1);
		//Gate 2
		weight++;
		bias++;
		gate2 = InnerState * *weight;
		gate2 += *bias;
		gate2 = Sig->CalculateMatrix(gate2);
		//Gate 3
		weight++;
		bias++;
		gate3 = InnerState * *weight;
		gate3 += *bias;
		gate3 = Tanh->CalculateMatrix(gate3);
		//Gate 4
		weight++;
		bias++;
		gate4 = InnerState * *weight;
		gate4 += *bias;
		gate4 = Sig->CalculateMatrix(gate4);
		//Update cell
		CellState.ElementwiseMultiply(gate1);
		gate3.ElementwiseMultiply(gate2);
		CellState += gate3;
		Tanh->CalculateInto(CellState, cellTanh);
		gate4.ElementwiseMultiply(cellTanh);
		InnerState.ReloadFromOther(gate4);
		for (size_t i = 0; i < CellStateSize; i++)
			tempOutput.SetValue(r, i, InnerState.GetValue(i));
	}

	std::vector<unsigned int> outputSize = Output.GetShape();
	Output = Tensor(tempOutput);
	Output.Reshape(outputSize);
}

Tensor& LSTM::GetOutput()
{
    return Output;
}

Tensor& LSTM::ComputeAndGetOutput()
{
    Compute();
    return Output;
}

void LSTM::GetBackwardPass(const Tensor& error, bool recursive)
{

}

void LSTM::UpdateWeightErrors(Matrix& gateIError, Matrix& gateRError, Matrix& inputTranspose, Matrix& dGate, Matrix& outputTranspose, int weight)
{

}

void LSTM::Train(Optimizer* optimizer)
{

}

void LSTM::SetTrainingMode(bool mode, bool recursive)
{
    TrainingMode = mode;
    InnerState.FillWith(0);
    CellState.FillWith(0);
    if (recursive && LayerInput)
        LayerInput->SetTrainingMode(mode, recursive);
}

Matrix& LSTM::GetWeight(unsigned char weight)
{
	std::list<Matrix>::iterator it = Weights.begin();
	for (unsigned char i = 0; i < weight; ++i)
		it++;
	return *it;
}

Matrix& LSTM::GetBias(unsigned char weight)
{
    std::list<Matrix>::iterator it = Biases.begin();
	for (unsigned char i = 0; i < weight; ++i)
		it++;
	return *it;
}

void LSTM::LoadFromJSON(const char* data, bool isFile)
{
    /*rapidjson::Document document;
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
    }*/
}

std::string LSTM::SaveToJSON(const char* fileName)
{
    /*rapidjson::Document doc;
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

    return std::string(buffer.GetString());*/

    return "";
}
