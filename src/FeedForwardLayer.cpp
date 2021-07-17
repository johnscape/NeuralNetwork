#include "FeedForwardLayer.h"
#include "Optimizer.h"

#if USE_GPU
#include "MatrixGPUMath.cuh"
#endif

FeedForwardLayer::FeedForwardLayer(Layer* inputLayer, unsigned int count) :
	Layer(inputLayer), Size(count), Weights(), Bias(1, count), InnerState(1, count),
	WeightError(), BiasError(1, count)
{
	unsigned int inputSize = 1;
	if (LayerInput)
		inputSize = LayerInput->GetOutput().GetVectorSize();
	Output.Reset(1, count);
	Weights.Reset(inputSize, count);
	WeightError.Reset(inputSize, count);
	LayerError.Reset(1, inputSize);
	function = &TanhFunction::GetInstance();

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
	if (input->GetOutput().GetVectorSize() == LayerInput->GetOutput().GetVectorSize())
		return;
	Weights.Reset(LayerInput->OutputSize(), Size);
}

void FeedForwardLayer::Compute()
{
	InnerState.FillWith(0);
	LayerInput->Compute();
	Matrix prev_out = LayerInput->GetOutput();
	InnerState = prev_out * Weights;
	InnerState += Bias;
	function->CalculateInto(InnerState, Output);
}

Matrix& FeedForwardLayer::ComputeAndGetOutput()
{
	Compute();
	return Output;
}

void FeedForwardLayer::SetActivationFunction(ActivationFunction* func)
{
	function = func;
}

void FeedForwardLayer::GetBackwardPass(const Matrix& error, bool recursive)
{
	Matrix derivate = function->CalculateDerivateMatrix(Output);
	LayerError.FillWith(0);
#if USE_GPU
	//GPUMath::FillWith(LayerError, 0); //do i even need this?
	derivate.CopyFromGPU();
#endif

	for (unsigned int neuron = 0; neuron < Size; neuron++)
	{
		float delta = error.GetValue(neuron);
		delta *= derivate.GetValue(neuron);
		for (unsigned int incoming = 0; incoming < LayerInput->GetOutput().GetVectorSize(); incoming++)
		{
			float wt = LayerInput->GetOutput().GetValue(incoming) * delta;
			WeightError.SetValue(incoming, neuron, wt);
			LayerError.AdjustValue(incoming, delta * Weights.GetValue(incoming, neuron));
		}

		BiasError.SetValue(neuron, delta);
	}

#if USE_GPU
	//WeightError.CopyToGPU();
#endif // USE_GPU

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

void FeedForwardLayer::LoadFromJSON(const char* data, bool isFile)
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

	unsigned int InputSize = 1;
	val = document["layer"]["size"];
	Size = val.GetUint();
	val = document["layer"]["inputSize"];
	
	unsigned int inputSize = val.GetUint();
	if (LayerInput)
		inputSize = LayerInput->GetOutput().GetVectorSize();
	Weights.Reset(inputSize, Size);
	Output.Reset(1, Size);
	Bias.Reset(1, Size);
	InnerState.Reset(1, Size);
	WeightError.Reset(inputSize, Size);
	LayerError.Reset(1, inputSize);
	BiasError.Reset(1, Size);

	rapidjson::StringBuffer buffer;
	rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);

	document["layer"]["weights"].Accept(writer);
	Weights.LoadFromJSON(buffer.GetString());

	buffer.Clear();
	writer.Reset(buffer);

	document["layer"]["bias"].Accept(writer);
	Bias.LoadFromJSON(buffer.GetString());*/
}

std::string FeedForwardLayer::SaveToJSON(const char* fileName)
{
	/*rapidjson::Document doc;
	doc.SetObject();

	rapidjson::Value layerSize, id, type, inputSize;
	layerSize.SetUint(Size);
	id.SetUint(Id);
	type.SetUint(1);
	if (LayerInput)
		inputSize.SetUint(LayerInput->GetOutput().GetVectorSize());
	else
		inputSize.SetUint(1);

	rapidjson::Document weight, bias;

	weight.Parse(Weights.SaveToJSON().c_str());
	bias.Parse(Bias.SaveToJSON().c_str());

	rapidjson::Value root(rapidjson::kObjectType);
	root.AddMember("id", id, doc.GetAllocator());
	root.AddMember("type", type, doc.GetAllocator());
	root.AddMember("size", layerSize, doc.GetAllocator());
	root.AddMember("inputSize", inputSize, doc.GetAllocator());
	root.AddMember("weights", weight, doc.GetAllocator());
	root.AddMember("bias", bias, doc.GetAllocator());

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
