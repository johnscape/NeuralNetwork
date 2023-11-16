#include <fstream>
#include "NeuralNetwork/Layers/RecurrentLayer.h"
#include "rapidjson/ostreamwrapper.h"
#include "rapidjson/writer.h"
#include "rapidjson/istreamwrapper.h"
#include "NeuralNetwork/Constants.h"

#if USE_GPU==USING_CUDA
#include "NeuralNetwork/CUDAMath.cuh"
#endif

RecurrentLayer::RecurrentLayer(Layer* inputLayer, unsigned int size, unsigned int timeSteps) :
	Layer(inputLayer), TimeSteps(timeSteps), CurrentStep(0), Size(size),
	Weights(), Bias(1, size), InnerState(), WeightError(), BiasError(1, size),
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
	for (unsigned int mat = 0; mat < InnerState.GetMatrixCount(); ++mat)
	{
		for (unsigned int row = 0; row < InnerState.GetShapeAt(0); ++row)
		{
			Matrix rowMat = InnerState.GetRowMatrix(mat, row);
			rowMat += Bias;
			rowMat += RecursiveState * RecursiveWeight;
			RecursiveState = rowMat;

            // TODO: Optimize this for CUDA
            RecursiveState.CopyFromGPU();
            InnerState.CopyFromGPU();


			for (unsigned int col = 0; col < InnerState.GetShapeAt(1); ++col)
			{
				unsigned int pos = mat * InnerState.GetShapeAt(0) * InnerState.GetShapeAt(1);
				pos += row * InnerState.GetShapeAt(1);
				pos += col;
				InnerState.SetValue(pos, RecursiveState.GetValue(col));
			}

            InnerState.CopyToGPU();
		}
	}

	if (!Output.IsSameShape(InnerState))
		Output = Tensor(InnerState);

    InnerState.CopyFromGPU();
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
	//TODO: Implement tensor elementwise multiply
	//LayerError = weight * (error .* derivate)
	TempMatrix errorMatrix = error.ToMatrixByRows();
	Tensor derivate = function->CalculateDerivateTensor(Output);
	LayerError = Tensor({(unsigned int)errorMatrix.GetRowCount(), LayerInput->OutputSize()}, nullptr);
#if USE_GPU
	derivate.CopyFromGPU();
#endif // USE_GPU

	TempMatrix states = InnerState.ToMatrixByRows();

	//If I call this function for once at every batch, this can stay, otherwise create a parameter
	std::vector<Matrix> powers;
	for (unsigned int i = 0; i < 3; ++i)
	{
		if (i == 0)
			powers.push_back(RecursiveWeight);
		else
			powers.push_back(RecursiveWeight.Power(i + 1));
	}

	for (unsigned int mat = 0; mat < error.GetMatrixCount(); ++mat)
	{
		for (unsigned int row = 0; row < error.GetShapeAt(0); ++row)
		{
			Matrix incoming = IncomingValues.GetRowMatrix(mat, row); //i_t
			incoming.Transpose();

			Matrix derivated = derivate.GetRowMatrix(mat, row); //o_t/s_t
			derivated.ElementwiseMultiply(error.GetRowMatrix(mat, row)); //E_t/s_t
			Matrix weightErr = incoming * derivated; //E_t/W

			for (int i = 0; i < TimeSteps; ++i) //Do I need to update for every timestep, or just for i=3?
			{
				if (i >= row)
					break;
				TempMatrix state = states.GetTempRowMatrix(row - i - 1);
				state.Transpose();
				if (i == 0)
					RecursiveWeightError += state * derivated;
				else
				{
					Matrix tmp = powers[i - 1] * state;
					tmp *= derivated;
					RecursiveWeightError += tmp;
				}

			}
			WeightError += weightErr;
			BiasError += derivated;
		}
	}

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

