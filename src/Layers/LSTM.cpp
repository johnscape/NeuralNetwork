#include "NeuralNetwork/Layers/LSTM.h"
#include "NeuralNetwork/Optimizers/Optimizer.h"
#include "NeuralNetwork/TempMatrix.h"

//NOTE Gate order is the following: Forget, Input, Activation, Output

LSTM::LSTM(Layer* inputLayer, unsigned int cellStateSize) :
    Layer(inputLayer), CellStateSize(cellStateSize), CellState(1, cellStateSize),
	InnerState(1, cellStateSize)
{
	for (unsigned char i = 0; i < 4; i++)
	{
		Matrix w(cellStateSize + inputLayer->OutputSize(), cellStateSize);
		Matrix b(1, cellStateSize);
		Weights.push_back(w);
		Biases.push_back(b);

		Matrix wcopy(w);
		Matrix bcopy(b);
		WeightErrors.push_back(wcopy);
		BiasErrors.push_back(bcopy);
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
    LSTM* r = new LSTM(LayerInput, CellStateSize);
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
	Tensor realInput = LayerInput->ComputeAndGetOutput();
	TempMatrix input = realInput.ToMatrixByRows();
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
		if (TrainingMode)
		{
			Gate1.push_back(gate1);
			Gate2.push_back(gate2);
			Gate3.push_back(gate3);
			Gate4.push_back(gate4);
		}
		//Update cell
		CellState.ElementwiseMultiply(gate1);
		gate3.ElementwiseMultiply(gate2);
		CellState += gate3;

		if (TrainingMode) //if we are training, we need to save the output of the output gate
		{
			SavedStates.push_back(InnerState);
			SavedCells.push_back(CellState);
		}

		Tanh->CalculateInto(CellState, cellTanh);
		gate4.ElementwiseMultiply(cellTanh);
		InnerState.ReloadFromOther(gate4);
		for (size_t i = 0; i < CellStateSize; i++)
			tempOutput.SetValue(r, i, InnerState.GetValue(i));
	}

	std::vector<unsigned int> outputSize = realInput.GetShape();
	outputSize[1] = CellStateSize;
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
	//get the errors from the next layer
	TempMatrix errors = error.ToMatrixByRows();
	Tensor derivate = Sig->CalculateDerivateTensor(Output);
	TempMatrix derivateMatrix = derivate.ToMatrixByRows();

	LayerError = Tensor({(unsigned int)errors.GetRowCount(), LayerInput->OutputSize()}, nullptr);

	//define the matrices to be used for delta calculations
	Matrix deltaOut(1, CellStateSize);
	Matrix deltaState(1, CellStateSize);
	Matrix deltaActivation(1, CellStateSize);
	Matrix deltaInput(1, CellStateSize);
	Matrix deltaForget(1, CellStateSize);
	Matrix deltaOutput(1, CellStateSize);
	Matrix lastDelta(1, CellStateSize);
	Matrix prevDeltaState(1, CellStateSize);

	//define and initialize iterators
	std::list<Matrix>::iterator state, cell, g1, g2, g3, g4;
	g1 = Gate1.end();
	g2 = Gate2.end();
	g3 = Gate3.end();
	g4 = Gate4.end();
	state = SavedStates.end();
	cell = SavedCells.end();

	g1--;
	g2--;
	g3--;
	g4--;
	state--;
	cell--;

	for (unsigned int step = derivateMatrix.GetRowCount() - 1; step >= 0 && step <= derivateMatrix.GetRowCount(); step--)
	{
		TempMatrix errorRow = errors.GetTempRowMatrix(step);
		TempMatrix derivateRow = errors.GetTempRowMatrix(step);
		Matrix cellTanh = Tanh->CalculateMatrix(*cell);

		deltaOut = errorRow + lastDelta;
		//delta state
		deltaState = Matrix::ElementwiseMultiply(deltaOut, *g4);
		deltaState.ElementwiseMultiply(Tanh->CalculateDerivateMatrix(cellTanh));
		deltaState += prevDeltaState;
		if (step < derivateMatrix.GetRowCount() - 1)
			deltaState.ElementwiseMultiply(*++g1--);
		//delta activation
		deltaActivation = Matrix::ElementwiseMultiply(deltaState, *g2);
		deltaActivation.ElementwiseMultiply(Tanh->CalculateDerivateMatrix(*g3));
		//delta input
		deltaInput = Matrix::ElementwiseMultiply(deltaState, *g3);
		deltaInput.ElementwiseMultiply(Sig->CalculateDerivateMatrix(*g2));
		//delta forget
		if (step > 0)
		{
			deltaForget = Matrix::ElementwiseMultiply(deltaState, *--cell++);
			deltaForget.ElementwiseMultiply(Sig->CalculateDerivateMatrix(*g1));
		}
		//delta output
		deltaOutput = Matrix::ElementwiseMultiply(deltaOut, cellTanh);
		deltaOutput.ElementwiseMultiply(Sig->CalculateDerivateMatrix(*g4));

		//calculate final deltas
		std::list<Matrix>::iterator weight, bias;
		weight = Weights.begin();

		deltaForget.Transpose();
		deltaInput.Transpose();
		deltaActivation.Transpose();
		deltaOutput.Transpose();
		Matrix combinedDelta(weight->GetRowCount(), deltaForget.GetColumnCount()); //TODO: Check if it's always a vector
		//Someone will kill me for this, but it looks better than repeating the same two lines
		combinedDelta += *weight * deltaForget; 		weight++;
		combinedDelta += *weight * deltaInput; 			weight++;
		combinedDelta += *weight * deltaActivation;		weight++;
		combinedDelta += *weight * deltaOutput;			weight++;

		combinedDelta.Transpose();
		Matrix inputDelta = combinedDelta.GetSubMatrix(0, CellStateSize, 1, LayerInput->OutputSize());
		lastDelta = combinedDelta.GetSubMatrix(0, 0, 1, CellStateSize);
		for (unsigned int i = 0; i < inputDelta.GetElementCount(); i++)
			LayerError.SetValue(step * LayerInput->OutputSize() + i, inputDelta.GetValue(i));

		//Modify weight errors
		weight = WeightErrors.begin();
		bias = BiasErrors.begin();

		Matrix tmp = deltaForget.OuterProduct(*state);
		tmp.Transpose();
		*weight += tmp;
		*bias += deltaForget;
		weight++; bias++;
		tmp = deltaInput.OuterProduct(*state);
		tmp.Transpose();
		*weight += tmp;
		*bias += deltaInput;
		weight++; bias++;
		tmp = deltaActivation.OuterProduct(*state);
		tmp.Transpose();
		*weight += tmp;
		*bias += deltaActivation;
		weight++; bias++;
		tmp = deltaOutput.OuterProduct(*state);
		tmp.Transpose();
		*weight += tmp;
		*bias += deltaOutput;

		g1--;
		g2--;
		g3--;
		g4--;
		state--;
		cell--;
	}

	if (recursive)
		LayerInput->GetBackwardPass(LayerError, true);
}

void LSTM::UpdateWeightErrors(Matrix& gateIError, Matrix& gateRError, Matrix& inputTranspose, Matrix& dGate, Matrix& outputTranspose, int weight)
{

}

void LSTM::Train(Optimizer* optimizer)
{
	std::list<Matrix>::iterator weight, weightErr, bias, biasErr;
	weight = Weights.begin();
	weightErr = WeightErrors.begin();
	bias = Biases.begin();
	biasErr = BiasErrors.begin();
	for (char i = 0; i < 4; ++i)
	{
		optimizer->ModifyWeights(*weight, *weightErr);
		optimizer->ModifyWeights(*bias, *biasErr);
		weightErr->FillWith(0);
		biasErr->FillWith(0);
	}

	Gate1.clear();
	Gate2.clear();
	Gate3.clear();
	Gate4.clear();

	SavedCells.clear();
	SavedStates.clear();
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

Matrix& LSTM::GetWeight(LSTM::Gate gate)
{
	if (gate == LSTM::Gate::FORGET)
		return GetWeight(0);
	else if (gate == LSTM::Gate::INPUT)
		return GetWeight(1);
	else if (gate == LSTM::Gate::ACTIVATION)
		return GetWeight(2);
	return GetWeight(3);
};

Matrix& LSTM::GetBias(LSTM::Gate gate)
{
	if (gate == LSTM::Gate::FORGET)
		return GetBias(0);
	else if (gate == LSTM::Gate::INPUT)
		return GetBias(1);
	else if (gate == LSTM::Gate::ACTIVATION)
		return GetBias(2);
	return GetBias(3);
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
