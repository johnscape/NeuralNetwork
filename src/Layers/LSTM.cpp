#include "NeuralNetwork/Layers/LSTM.h"
#include "NeuralNetwork/Optimizers/Optimizer.h"
#include "NeuralNetwork/TempMatrix.h"

//NOTE Gate order is the following: Forget, Input, Activation, Output

LSTM::LSTM(Layer* inputLayer, unsigned int cellStateSize) :
    Layer(inputLayer), CellStateSize(cellStateSize), CellState(1, cellStateSize), InputSize(inputLayer->OutputSize()),
	InnerState(1, cellStateSize + inputLayer->OutputSize())
{
    ForgetGateWeight.Reset(InnerState.GetColumnCount(), CellStateSize);
    ForgetGateBias.Reset(1, CellStateSize);
    ForgetGateWeightError.Reset(InnerState.GetColumnCount(), CellStateSize);
    ForgetGateBiasError.Reset(1, CellStateSize);

    InputGateSigmoidWeight.Reset(InnerState.GetColumnCount(), CellStateSize);
    InputGateSigmoidBias.Reset(1, CellStateSize);
    InputGateSigmoidWeightError.Reset(InnerState.GetColumnCount(), CellStateSize);
    InputGateSigmoidBiasError.Reset(1, CellStateSize);

    InputGateTanhWeight.Reset(InnerState.GetColumnCount(), CellStateSize);
    InputGateTanhBias.Reset(1, CellStateSize);
    InputGateTanhWeightError.Reset(InnerState.GetColumnCount(), CellStateSize);
    InputGateTanhBiasError.Reset(1, CellStateSize);

    OutputGateWeight.Reset(InnerState.GetColumnCount(), CellStateSize);
    OutputGateBias.Reset(1, CellStateSize);
    OutputGateWeightError.Reset(InnerState.GetColumnCount(), CellStateSize);
    OutputGateBiasError.Reset(1, CellStateSize);

    Tanh = &TanhFunction::GetInstance();
    Sig = &Sigmoid::GetInstance();
}

LSTM::~LSTM()
{
}

Layer* LSTM::Clone()
{
    LSTM* r = new LSTM(LayerInput, CellStateSize);

	r->ForgetGateWeight.Copy(ForgetGateWeight);
    r->ForgetGateBias.Copy(ForgetGateBias);

    r->InputGateSigmoidWeight.Copy(InputGateSigmoidWeight);
    r->InputGateSigmoidBias.Copy(InputGateSigmoidBias);

    r->InputGateTanhWeight.Copy(InputGateTanhWeight);
    r->InputGateTanhBias.Copy(InputGateTanhBias);

    r->OutputGateWeight.Copy(OutputGateWeight);
    r->OutputGateBias.Copy(OutputGateBias);

    return r;
}

void LSTM::Compute()
{

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

	if (recursive)
		LayerInput->GetBackwardPass(LayerError, true);
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

Matrix& LSTM::GetWeight(LSTM::Gate gate)
{
	if (gate == LSTM::Gate::FORGET)
		return ForgetGateWeight;
	else if (gate == LSTM::Gate::INPUT)
		return InputGateSigmoidWeight;
	else if (gate == LSTM::Gate::ACTIVATION)
		return InputGateTanhWeight;
	return OutputGateWeight;
};

Matrix& LSTM::GetBias(LSTM::Gate gate)
{
	if (gate == LSTM::Gate::FORGET)
		return ForgetGateBias;
	else if (gate == LSTM::Gate::INPUT)
		return InputGateSigmoidBias;
	else if (gate == LSTM::Gate::ACTIVATION)
		return InputGateTanhBias;
	return OutputGateBias;
}

void LSTM::LoadFromJSON(rapidjson::Value& jsonData)
{
	/*if (jsonData.HasMember("layer"))
		jsonData = jsonData["layer"];
	if (jsonData["type"].GetUint64() != static_cast<unsigned int>(Layer::LayerType::LSTM))
		throw LayerTypeException();

	CellStateSize = jsonData["size"].GetUint64();
	Id = jsonData["id"].GetUint64();
	rapidjson::Value weight, bias;
	weight = jsonData["weights"];
	bias = jsonData["biases"];
	std::list<Matrix>::iterator weightIt, biasIt, weightErrIt, biasErrIt;
	weightIt = Weights.begin();
	biasIt = Biases.begin();
	weightErrIt = WeightErrors.begin();
	biasErrIt = BiasErrors.begin();

	for (unsigned char i = 0; i < 4; i++)
	{
		weightIt->LoadFromJSON(weight[i]);
		biasIt->LoadFromJSON(bias[i]);

		weightErrIt->Reset(weightIt->GetRowCount(), weightIt->GetColumnCount());
		biasErrIt->Reset(biasIt->GetRowCount(), biasIt->GetColumnCount());
		weightIt++;
		biasIt++;
		weightErrIt++;
		biasErrIt++;
	}*/

}

rapidjson::Value LSTM::SaveToJSONObject(rapidjson::Document& document) const
{
	rapidjson::Value layer(rapidjson::kObjectType);
	/*rapidjson::Value type, id, size;

	type.SetUint64(static_cast<unsigned int>(Layer::LayerType::LSTM));
	id.SetUint64(Id);
	size.SetUint64(CellStateSize);

	rapidjson::Value weights, biases;
	weights.SetArray();
	biases.SetArray();

	std::list<Matrix>::const_iterator weightIt, biasIt;
	weightIt = Weights.begin();
	biasIt = Biases.begin();
	for (unsigned char i = 0; i < 4; ++i)
	{
		rapidjson::Value tmp;
		tmp = weightIt->SaveToJSONObject(document);
		weights.PushBack(tmp, document.GetAllocator());
		tmp = biasIt->SaveToJSONObject(document);
		biases.PushBack(tmp, document.GetAllocator());
		weightIt++;
		biasIt++;
	}

	layer.AddMember("id", id, document.GetAllocator());
	layer.AddMember("type", type, document.GetAllocator());
	layer.AddMember("size", size, document.GetAllocator());
	layer.AddMember("weights", weights, document.GetAllocator());
	layer.AddMember("biases", biases, document.GetAllocator());*/

	return layer;
}
