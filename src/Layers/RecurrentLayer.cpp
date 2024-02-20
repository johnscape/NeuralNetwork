#include <fstream>
#include "NeuralNetwork/Layers/RecurrentLayer.h"
#include "rapidjson/ostreamwrapper.h"
#include "rapidjson/writer.h"
#include "rapidjson/istreamwrapper.h"
#include "NeuralNetwork/Constants.h"

#if USE_GPU==USING_CUDA
#include "NeuralNetwork/CUDAFunctions.cuh"
#endif

RecurrentLayer::RecurrentLayer(Layer* inputLayer, unsigned int size, unsigned int timeSteps) :
	Layer(inputLayer), TimeSteps(timeSteps), CurrentStep(0), Size(size),
	Weights(), Bias(1, size), InnerState(), WeightError(), BiasError(1, size), InnerRow(1, size),
	RecursiveWeight(size, size), RecursiveWeightError(size, size), RecursiveState(1, size)
{
	Weights.Reset(inputLayer->GetOutput().GetShapeAt(1), size);
	Output = Tensor({1, size}, nullptr);
	WeightError.Reset(inputLayer->GetOutput().GetShapeAt(1), size);
	function = &TanhFunction::GetInstance();

	Bias.FillWith(1);
	Weights.FillWithRandom();
	RecursiveWeight.FillWithRandom();
}

RecurrentLayer::~RecurrentLayer()
{
}

Layer* RecurrentLayer::Clone()
{
	RecurrentLayer* r = new RecurrentLayer(LayerInput, Size, TimeSteps);
	r->GetWeights().Copy(Weights);
	r->GetRecurrentWeights().Copy(RecursiveWeight);
	r->GetBias().Copy(Bias);
	return r;
}

void RecurrentLayer::Compute()
{
	IncomingValues = LayerInput->ComputeAndGetOutput();
	InnerState = IncomingValues * Weights;
    unsigned int elementPointer = 0;
    while (elementPointer < InnerState.GetElementCount())
    {
        InnerState.CopyPartTo(InnerRow, elementPointer, 0, InnerRow.GetColumnCount());
        InnerRow += Bias;
        RecursiveState *= RecursiveWeight;
        RecursiveState += InnerRow;
        RecursiveState.CopyPartTo(InnerState, 0, elementPointer, RecursiveState.GetColumnCount());

        elementPointer += Size;
    }

	if (!Output.IsSameShape(InnerState))
		Output = Tensor(InnerState);
    Output.CopyFromGPU();

	function->CalculateInto(InnerState, Output);
}

Tensor& RecurrentLayer::GetOutput()
{
	return Output;
}

Tensor& RecurrentLayer::ComputeAndGetOutput()
{
	Compute();
	return Output;
}

void RecurrentLayer::SetActivationFunction(ActivationFunction* func)
{
	function = func;
}

void RecurrentLayer::GetBackwardPass(const Tensor& error, bool recursive)
{
    Tensor derivates = function->CalculateDerivateTensor(Output);
    TempMatrix inputs = IncomingValues.ToMatrixByRows();
    TempMatrix states = InnerState.ToMatrixByRows();
    TempMatrix outputs = derivates.ToMatrixByRows();
    TempMatrix errors = error.ToMatrixByRows();

    Matrix recursiveWeightPower(RecursiveWeight);
    Matrix delta(1, Size);

    LayerError = Tensor({inputs.GetRowCount(), Weights.GetRowCount()}, nullptr);
    Matrix LayerErrorRow( Weights.GetRowCount(), 1);

    unsigned char currentPower = TimeSteps;
    recursiveWeightPower = RecursiveWeight.Power(currentPower);


    for (unsigned int timeStep = outputs.GetRowCount() - 1; timeStep > 0; timeStep--)
    {
        delta.FillWith(1);

        TempMatrix errorRow = errors.GetTempRowMatrix(timeStep);
        TempMatrix derivateRow = outputs.GetTempRowMatrix(timeStep);

        delta.ElementwiseMultiply(errorRow);
        delta.ElementwiseMultiply(derivateRow);

        // Bias
        BiasError += delta;
        delta.Transpose();

        // Input weights
        WeightError += delta * inputs.GetTempRowMatrix(timeStep);

        // Recurrent weights
        if (currentPower > timeStep)
        {
            currentPower--;
            recursiveWeightPower = RecursiveWeight.Power(currentPower);
        }

        // TODO: Check if order works
        RecursiveWeightError += recursiveWeightPower * delta * states.GetTempRowMatrix(timeStep - currentPower);

        // Layer error
        LayerErrorRow = Weights * delta;
        LayerErrorRow.CopyPartTo(LayerError, 0, timeStep * Weights.GetRowCount(), Weights.GetRowCount());

        delta.Transpose();

    }

    BiasError *= (1 / (float)outputs.GetRowCount());
    WeightError *= (1 / (float)outputs.GetRowCount());
    RecursiveWeightError *= (1 / (float)outputs.GetRowCount());

	if (recursive)
		LayerInput->GetBackwardPass(LayerError);
}

void RecurrentLayer::Train(Optimizer* optimizer)
{
	optimizer->ModifyWeights(Weights, WeightError);
	optimizer->ModifyWeights(RecursiveWeight, RecursiveWeightError);
	optimizer->ModifyWeights(Bias, BiasError);

	WeightError.FillWith(0);
	RecursiveWeightError.FillWith(0);
	BiasError.FillWith(0);

#if USE_GPU
	Weights.CopyToGPU();
	RecursiveWeight.CopyToGPU();
	Bias.CopyToGPU();
#endif // USE_GPU

}

void RecurrentLayer::SetTrainingMode(bool mode)
{
	TrainingMode = mode;
}

Matrix& RecurrentLayer::GetWeights()
{
	return Weights;
}

Matrix& RecurrentLayer::GetBias()
{
	return Bias;
}

Matrix& RecurrentLayer::GetRecurrentWeights()
{
	return RecursiveWeight;
}

void RecurrentLayer::LoadFromJSON(rapidjson::Value& jsonData)
{
	if (jsonData.HasMember("layer"))
		jsonData = jsonData["layer"];
	if (jsonData["type"].GetUint64() != static_cast<unsigned int>(Layer::LayerType::RECURRENT))
		throw LayerTypeException();

	Id = jsonData["id"].GetUint64();
	Size = jsonData["size"].GetUint64();
	function = ActivationFunctionLibrary::GetActivationFunction(
			static_cast<ActivationFunctionType>(jsonData["activation"].GetUint64())
			);

	rapidjson::Value tmpValue;
	tmpValue = jsonData["weight"];
	Weights.LoadFromJSON(tmpValue);
	tmpValue = jsonData["recurrent"];
	RecursiveWeight.LoadFromJSON(tmpValue);
	tmpValue = jsonData["bias"];
	Bias.LoadFromJSON(tmpValue);

	WeightError.Reset(Weights.GetRowCount(), Weights.GetColumnCount());
	RecursiveWeightError.Reset(Size, Size);
	BiasError.Reset(Bias.GetRowCount(), Bias.GetColumnCount());
}

rapidjson::Value RecurrentLayer::SaveToJSONObject(rapidjson::Document& document) const
{
	rapidjson::Value layer(rapidjson::kObjectType);
	rapidjson::Value type, id, size, activation;

	type.SetUint64(static_cast<unsigned int>(Layer::LayerType::RECURRENT));
	id.SetUint64(Id);
	size.SetUint64(Size);
	activation.SetUint64(static_cast<unsigned int>(function->GetActivationFunctionType()));

	rapidjson::Value weights(rapidjson::kObjectType);
	weights.AddMember("matrix", Weights.SaveToJSONObject(document), document.GetAllocator());
	rapidjson::Value bias(rapidjson::kObjectType);
	bias.AddMember("matrix", Bias.SaveToJSONObject(document), document.GetAllocator());
	rapidjson::Value recurrentWeights(rapidjson::kObjectType);
	recurrentWeights.AddMember("matrix", RecursiveWeight.SaveToJSONObject(document), document.GetAllocator());

	layer.AddMember("id", id, document.GetAllocator());
	layer.AddMember("type", type, document.GetAllocator());
	layer.AddMember("size", size, document.GetAllocator());
	layer.AddMember("activation", activation, document.GetAllocator());
	layer.AddMember("weight", weights, document.GetAllocator());
	layer.AddMember("recurrent", recurrentWeights, document.GetAllocator());
	layer.AddMember("bias", bias, document.GetAllocator());

	return layer;
}

