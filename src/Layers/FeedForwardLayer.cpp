#include "NeuralNetwork/Layers/FeedForwardLayer.h"
#include "NeuralNetwork/Optimizers/Optimizer.h"
#include "NeuralNetwork/Constants.h"

#if USE_GPU==USING_CUDA
#include "NeuralNetwork/CUDAFunctions.cuh"
#endif

FeedForwardLayer::FeedForwardLayer(Layer* inputLayer, unsigned int count) :
	Layer(inputLayer), Size(count), Weights(), Bias(1, count), InnerState({1, count}, nullptr),
	WeightError(), BiasError(1, count)
{
	unsigned int inputSize = 1;
	if (LayerInput)
		inputSize = LayerInput->GetOutput().GetShapeAt(1);
	Output = Tensor({1, count}, nullptr);
	Weights.Reset(inputSize, count);
	WeightError.Reset(inputSize, count);
	function = &Sigmoid::GetInstance();

	Bias.FillWith(1);
	Weights.FillWithRandom();
}

Layer* FeedForwardLayer::Clone()
{
	FeedForwardLayer* l = new FeedForwardLayer(LayerInput, Size);

	l->GetWeights().Copy(Weights);
	l->GetOutput().Copy(Output);
	l->GetBias().Copy(Bias);

	l->SetActivationFunction(function);
	return l;
}

FeedForwardLayer::~FeedForwardLayer()
{
}

void FeedForwardLayer::SetInput(Layer* input)
{
	if (input == LayerInput)
		return;
	LayerInput = input;
	if (input->GetOutput().GetShapeAt(1) == LayerInput->GetOutput().GetShape()[1])
		return;
	Weights.Reset(LayerInput->OutputSize(), Size);
}

void FeedForwardLayer::Compute()
{
	Tensor prev_out = LayerInput->ComputeAndGetOutput();

	InnerState = prev_out * Weights;
	InnerState += Bias;
	if(!Output.IsSameShape(InnerState))
		Output = Tensor(InnerState.GetShape());
	function->CalculateInto(InnerState, Output);
}

Tensor& FeedForwardLayer::ComputeAndGetOutput()
{
	Compute();
	return Output;
}

void FeedForwardLayer::SetActivationFunction(ActivationFunction* func)
{
	function = func;
}

void FeedForwardLayer::GetBackwardPass(const Tensor& error, bool recursive)
{
    Tensor derivates = function->CalculateDerivateTensor(Output);
    TempMatrix derivateRows = derivates.ToMatrixByRows();
    TempMatrix errorRows = error.ToMatrixByRows();
    TempMatrix inputRows = LayerInput->GetOutput().ToMatrixByRows();

    LayerError = Tensor(LayerInput->GetOutput().GetShape());

    Matrix delta(errorRows.GetRowCount(), Size);
    Matrix layerErrorRow;
    delta.FillWith(1);

    delta.ElementwiseMultiply(errorRows);
    delta.ElementwiseMultiply(derivateRows);

    for (unsigned int i = 0; i < delta.GetRowCount(); i++)
    {
        TempMatrix deltaRow = delta.GetTempRowMatrix(i);
        BiasError += deltaRow;

        deltaRow.Transpose();
        TempMatrix inputRow = inputRows.GetTempRowMatrix(i);
        WeightError += deltaRow * inputRow;

        deltaRow.Transpose();
        layerErrorRow = delta * Weights;
        layerErrorRow.CopyPartTo(LayerError, 0, i * Weights.GetRowCount(), Weights.GetRowCount());
    }

	if (recursive)
		LayerInput->GetBackwardPass(LayerError);
}

void FeedForwardLayer::Train(Optimizer* optimizer)
{
	optimizer->ModifyWeights(Weights, WeightError);
	optimizer->ModifyWeights(Bias, BiasError);

	WeightError.FillWith(0);
	BiasError.FillWith(0);

#if USE_GPU
	Weights.CopyToGPU();
	Bias.CopyToGPU();
#endif // USE_GPU

}

Matrix& FeedForwardLayer::GetBias()
{
	return Bias;
}

Matrix& FeedForwardLayer::GetWeights()
{
	return Weights;
}

void FeedForwardLayer::LoadFromJSON(rapidjson::Value& jsonData)
{
	if (jsonData.HasMember("layer"))
		jsonData = jsonData["layer"];
	if (jsonData["type"] != static_cast<unsigned int>(Layer::LayerType::FORWARD))
		throw LayerTypeException();
	Id = jsonData["id"].GetUint64();
	Size = jsonData["size"].GetUint64();
	function = ActivationFunctionLibrary::GetActivationFunction(
			static_cast<ActivationFunctionType>(jsonData["activation"].GetUint64())
			);
	rapidjson::Value tmpValue;
	tmpValue = jsonData["weight"];
	Weights.LoadFromJSON(tmpValue);
	tmpValue = jsonData["bias"];
	Bias.LoadFromJSON(tmpValue);

	Output = Tensor({1, Size}, nullptr);
	WeightError.Reset(Weights.GetRowCount(), Weights.GetColumnCount());
	BiasError.Reset(Bias.GetRowCount(), Bias.GetColumnCount());
}

rapidjson::Value FeedForwardLayer::SaveToJSONObject(rapidjson::Document &document) const
{
	rapidjson::Value layer(rapidjson::kObjectType);
	rapidjson::Value type, id, size, activation;

	type.SetUint64(static_cast<unsigned int>(Layer::LayerType::FORWARD));
	id.SetUint64(Id);
	size.SetUint64(Size);
	activation.SetUint64(static_cast<unsigned int>(function->GetActivationFunctionType()));

	rapidjson::Value weights(rapidjson::kObjectType);
	weights.AddMember("matrix", Weights.SaveToJSONObject(document), document.GetAllocator());
	rapidjson::Value bias(rapidjson::kObjectType);
	bias.AddMember("matrix", Bias.SaveToJSONObject(document), document.GetAllocator());

	layer.AddMember("id", id, document.GetAllocator());
	layer.AddMember("type", type, document.GetAllocator());
	layer.AddMember("size", size, document.GetAllocator());
	layer.AddMember("activation", activation, document.GetAllocator());
	layer.AddMember("weight", weights, document.GetAllocator());
	layer.AddMember("bias", bias, document.GetAllocator());

	return layer;
}
