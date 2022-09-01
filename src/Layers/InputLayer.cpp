#include <fstream>
#include "NeuralNetwork/Layers/InputLayer.h"
#include "NeuralNetwork/Constants.h"
#include "NeuralNetwork/TensorException.hpp"
#include "rapidjson/ostreamwrapper.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/istreamwrapper.h"

InputLayer::InputLayer(unsigned int size) : Layer(), Size(1, size)
{
	LayerInput = nullptr;
	Output = Tensor({1, size}, nullptr);
}

InputLayer::InputLayer(std::vector<unsigned int> size) : Layer(), Size(size)
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
	rapidjson::Document document;
	if (!isFile)
		document.Parse(data);
	else
	{
		std::ifstream r(data);
		rapidjson::IStreamWrapper isw(r);
		document.ParseStream(isw);
	}

	LoadFromJSON(document);
}

void InputLayer::LoadFromJSON(rapidjson::Value& jsonData)
{
	Size.clear();
	if (jsonData.HasMember("layer"))
		jsonData = jsonData["layer"];
	if (jsonData["type"].GetUint64() != static_cast<unsigned int>(Layer::LayerType::INPUT))
		throw LayerTypeException();
	Id = jsonData["id"].GetUint64();
	jsonData = jsonData["size"];
	for (rapidjson::Value::ConstValueIterator itr = jsonData.Begin(); itr != jsonData.End(); itr++)
		Size.push_back(itr->GetUint64());
}

std::string InputLayer::SaveToJSON(const char* fileName) const
{
	rapidjson::Document doc;
	doc.SetObject();

	rapidjson::Value root = SaveToJSONObject(doc);

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

rapidjson::Value InputLayer::SaveToJSONObject(rapidjson::Document& document) const
{
	rapidjson::Value id, type, inputSize;
	rapidjson::Value layer(rapidjson::kObjectType);

	id.SetUint64(Id);
	type.SetUint64(static_cast<unsigned int>(Layer::LayerType::INPUT));
	inputSize.SetArray();
	for (unsigned int i : Size)
		inputSize.PushBack(i, document.GetAllocator());

	layer.AddMember("id", id, document.GetAllocator());
	layer.AddMember("type", type, document.GetAllocator());
	layer.AddMember("size", inputSize, document.GetAllocator());

	return layer;
}

