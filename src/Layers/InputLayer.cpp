#include "NeuralNetwork/Layers/InputLayer.h"
#include "NeuralNetwork/Layers/LayerException.hpp"
#include "NeuralNetwork/Constants.h"
#include "NeuralNetwork/TensorException.hpp"

InputLayer::InputLayer(unsigned int size) : Layer(), Size(1, size)
{
	LayerInput = nullptr;
	Output = Tensor({1, size}, nullptr);
}

InputLayer::InputLayer(std::vector<unsigned int>& size) : Layer(), Size(size)
{
	LayerInput = nullptr;
	Output = Tensor(Size, nullptr);
}

Layer* InputLayer::Clone()
{
	return new InputLayer(Size);
}

void InputLayer::Compute()
{}

Tensor& InputLayer::ComputeAndGetOutput()
{
	return Output;
}

void InputLayer::SetInput(const Tensor& input)
{
	Output.ReloadFromOther(input);
}

void InputLayer::SetInput(const Matrix &input)
{
	SetInput(Tensor(input));
}

void InputLayer::GetBackwardPass(const Tensor& error, bool recursive)
{
}

void InputLayer::LoadFromJSON(const char* data, bool isFile)
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
	val = document["layer"]["size"];
	Size = val.GetUint();*/
}

std::string InputLayer::SaveToJSON(const char* fileName)
{
	/*rapidjson::Document doc;
	doc.SetObject();

	rapidjson::Value input, id, type;
	input.SetUint(Size);
	id.SetUint(Id);
	type.SetUint(0);

	rapidjson::Value root(rapidjson::kObjectType);
	root.AddMember("id", id, doc.GetAllocator());
	root.AddMember("type", type, doc.GetAllocator());
	root.AddMember("size", input, doc.GetAllocator());

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

