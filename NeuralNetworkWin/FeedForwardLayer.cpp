#include "FeedForwardLayer.h"
#include "Optimizer.h"

FeedForwardLayer::FeedForwardLayer(Layer* inputLayer, unsigned int count) : Layer(inputLayer), Size(count)
{
	unsigned int inputSize = 1;
	if (LayerInput)
		inputSize = LayerInput->GetOutput()->GetVectorSize();
	Weights = new Matrix(inputSize, count);
	Output = new Matrix(1, count);
	Bias = new Matrix(1, count);
	InnerState = new Matrix(1, count);
	WeightError = new Matrix(inputSize, count);
	LayerError = new Matrix(1, inputSize);
	BiasError = new Matrix(1, count);
	function = new TanhFunction();

	MatrixMath::FillWith(Bias, 1);
	MatrixMath::FillWithRandom(Weights);
}

Layer* FeedForwardLayer::Clone()
{
	FeedForwardLayer* l = new FeedForwardLayer(LayerInput, Size);

	MatrixMath::Copy(Weights, l->GetWeights());
	MatrixMath::Copy(Output, l->GetOutput());
	MatrixMath::Copy(Bias, l->GetBias());

	l->SetActivationFunction(function);
	return l;
}

FeedForwardLayer::~FeedForwardLayer()
{
	if (function)
		delete function;
	delete Weights;
	delete Bias;
	delete InnerState;

	delete BiasError;
	delete WeightError;
}

void FeedForwardLayer::SetInput(Layer* input)
{
	LayerInput = input;
	delete Weights;
	Weights = new Matrix(LayerInput->OutputSize(), Size);
}

void FeedForwardLayer::Compute()
{
	MatrixMath::FillWith(InnerState, 0);
	LayerInput->Compute();
	Matrix* prev_out = LayerInput->GetOutput();
	MatrixMath::Multiply(prev_out, Weights, InnerState);
	MatrixMath::AddIn(InnerState, Bias);
	function->CalculateInto(InnerState, Output);
}

Matrix* FeedForwardLayer::ComputeAndGetOutput()
{
	Compute();
	return Output;
}

void FeedForwardLayer::SetActivationFunction(ActivationFunction* func)
{
	if (function)
		delete function;
	function = func;
}

void FeedForwardLayer::GetBackwardPass(Matrix* error, bool recursive)
{
	Matrix* derivate = function->CalculateDerivateMatrix(Output);
	MatrixMath::FillWith(LayerError, 0);

	for (unsigned int neuron = 0; neuron < Size; neuron++)
	{
		float delta = error->GetValue(neuron);
		delta *= derivate->GetValue(neuron);
		for (unsigned int incoming = 0; incoming < LayerInput->GetOutput()->GetVectorSize(); incoming++)
		{
			float wt = LayerInput->GetOutput()->GetValue(incoming) * delta;
			WeightError->SetValue(incoming, neuron, wt);
			LayerError->AdjustValue(incoming, delta * Weights->GetValue(incoming, neuron));
		}

		BiasError->SetValue(neuron, delta);
	}

	delete derivate;

	if (recursive)
		LayerInput->GetBackwardPass(LayerError);
}

void FeedForwardLayer::Train(Optimizer* optimizer)
{
	optimizer->ModifyWeights(Weights, WeightError);
	optimizer->ModifyWeights(Bias, BiasError);

	MatrixMath::FillWith(WeightError, 0);
	MatrixMath::FillWith(BiasError, 0);
}

Matrix* FeedForwardLayer::GetBias()
{
	return Bias;
}

Matrix* FeedForwardLayer::GetWeights()
{
	return Weights;
}

void FeedForwardLayer::LoadFromJSON(const char* data, bool isFile)
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

	delete Weights;
	delete Output;
	delete InnerState;
	delete WeightError;
	delete LayerError;
	delete Bias;
	delete BiasError;

	unsigned int InputSize = 1;
	val = document["layer"]["size"];
	Size = val.GetUint();
	val = document["layer"]["inputSize"];
	
	unsigned int inputSize = val.GetUint();
	if (LayerInput)
		inputSize = LayerInput->GetOutput()->GetVectorSize();
	Weights = new Matrix(inputSize, Size);
	Output = new Matrix(1, Size);
	Bias = new Matrix(1, Size);
	InnerState = new Matrix(1, Size);
	WeightError = new Matrix(inputSize, Size);
	LayerError = new Matrix(1, inputSize);
	BiasError = new Matrix(1, Size);

	rapidjson::StringBuffer buffer;
	rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);

	document["layer"]["weights"].Accept(writer);
	Weights->LoadFromJSON(buffer.GetString());

	buffer.Clear();
	writer.Reset(buffer);

	document["layer"]["bias"].Accept(writer);
	Bias->LoadFromJSON(buffer.GetString());

	
}

std::string FeedForwardLayer::SaveToJSON(const char* fileName)
{
	rapidjson::Document doc;
	doc.SetObject();

	rapidjson::Value layerSize, id, type, inputSize;
	layerSize.SetUint(Size);
	id.SetUint(Id);
	type.SetUint(1);
	if (LayerInput)
		inputSize.SetUint(LayerInput->GetOutput()->GetVectorSize());
	else
		inputSize.SetUint(1);

	rapidjson::Document weight, bias;

	weight.Parse(Weights->SaveToJSON().c_str());
	bias.Parse(Bias->SaveToJSON().c_str());

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

	return std::string(buffer.GetString());

}
