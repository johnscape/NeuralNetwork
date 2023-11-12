#include <fstream>
#include <utility>
#include "NeuralNetwork/Layers/InputLayer.h"
#include "NeuralNetwork/Constants.h"
#include "NeuralNetwork/TensorException.hpp"

InputLayer::InputLayer(unsigned int size) : Layer(), Size(1, size)
{
	LayerInput = nullptr;
	Output = Tensor({1, size}, nullptr);
}

InputLayer::InputLayer(std::vector<unsigned int> size) : Layer(), Size(std::move(size))
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

