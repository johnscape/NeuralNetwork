#include "NeuralNetwork/Model.h"
#include "NeuralNetwork/Layers/Layer.h"
#include "NeuralNetwork/Layers/InputLayer.h"
#include <map>
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/writer.h"
#include "rapidjson/ostreamwrapper.h"
#include <fstream>
#include <sstream>

Model::Model() : inputLayer(nullptr), outputLayer(nullptr)
{
}

Model::Model(const Model& m) : inputLayer(nullptr), outputLayer(nullptr)
{
	CopyFromModel(m);
}

Model& Model::operator=(const Model& other)
{
	Layers.clear();

	CopyFromModel(other);

	return *this;
}

Model::~Model()
{
	std::list<Layer*>::const_iterator layerIterator;
	std::list<bool>::const_iterator deleteIterator;

	layerIterator = Layers.begin();
	deleteIterator = ToDelete.begin();

	while (layerIterator != Layers.end())
	{
		if (*deleteIterator)
			delete *layerIterator;
		deleteIterator++;
		layerIterator++;
	}

	Layers.clear();
}

void Model::AddLayer(Layer* layer, bool toDelete)
{
	Layers.push_back(layer);
	ToDelete.push_back(toDelete);
	UpdateInputOutput();
}

Layer* Model::GetLayer(unsigned int id)
{
	return FindLayerWithId(id);
}

unsigned int Model::GetLayerCount() const
{
	return Layers.size();
}

void Model::SaveModel(const char* fileName) const
{
	rapidjson::Document document = SaveToDocument();

	std::ofstream w(fileName);
	rapidjson::OStreamWrapper osw(w);
	rapidjson::PrettyWriter<rapidjson::OStreamWrapper> writer(osw);
	document.Accept(writer);
	w.close();
}

void Model::LoadModel(const char* fileName)
{
	std::ifstream reader(fileName);
	if (!reader.good())
		return; //TODO: Throw exception
	std::stringstream buffer;
	buffer << reader.rdbuf();
	LoadFromString(buffer.str());
}

std::string Model::SaveToString() const
{

	rapidjson::Document document = SaveToDocument();
	rapidjson::StringBuffer buffer;
	rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
	document.Accept(writer);

	return std::string(buffer.GetString());
}

void Model::LoadFromString(const std::string& json)
{
	LoadFromString(json.c_str());
}

void Model::LoadFromString(const char* json)
{
	rapidjson::Document document;
	document.Parse(json);

	if (document.MemberBegin() == document.MemberEnd() || !document.HasMember("model"))
		return; //TODO: Throw error

	rapidjson::Value value;
	value = document["model"]["layers"];
	for (rapidjson::Value::ValueIterator itr = value.Begin(); itr != value.End(); itr++)
	{
		rapidjson::Value& layer = *itr;
		if (!layer.HasMember("layer"))
			return; //TODO: Throw error
		layer = layer["layer"];
		Layer::LayerType type = static_cast<Layer::LayerType>(layer["type"].GetUint64());
		Layer* newLayer = Layer::Create(type, {0}, nullptr);
		newLayer->LoadFromJSON(layer);
		Layers.push_back(newLayer);
		ToDelete.push_back(true);
	}
}

Tensor Model::Compute(const Tensor& input) const
{
	if (inputLayer == nullptr || outputLayer == nullptr)
		throw LayerTypeException();
	inputLayer->SetInput(input);
	Tensor result = outputLayer->ComputeAndGetOutput();
	return result;
}

Layer* Model::GetLastLayer() const
{
	return Layers.back();
}

Layer* Model::GetOutput() const
{
	return outputLayer;
}

Layer* Model::GetInput() const
{
	return inputLayer;
}

unsigned int Model::LayerCount() const
{
	return Layers.size();
}

Layer* Model::GetLayerAt(unsigned int n) const
{
	if (n >= Layers.size())
		return nullptr;
	std::list<Layer*>::const_iterator layer;
	layer = Layers.begin();
	for (unsigned int i = 0; i < n; i++)
		layer++;
	return *layer;
}

void Model::FindOutput()
{
	//the output is, which is not used by any other layer as input
	std::list<Layer*>::iterator first, second;

	for (first = Layers.begin(); first != Layers.end(); first++)
	{
		bool braked = false;
		for (second = Layers.begin(); second != Layers.end(); second++)
		{
			if (first == second)
				continue;
			if ((*second)->GetInputLayer() == (*first))
			{
				braked = true;
				break;
			}
		}

		if (!braked)
		{
			outputLayer = *first;
			return;
		}
	}
}

void Model::FindInput()
{
	std::list<Layer*>::iterator it = Layers.begin();
	while (it != Layers.end())
	{
		Layer* currentLayer = *it;
		if (dynamic_cast<InputLayer*>(currentLayer) != nullptr)
		{
			inputLayer = currentLayer;
			return;
		}
		it++;
	}
	inputLayer = nullptr;
}

Layer* Model::FindLayerWithId(unsigned int id)
{
	for (std::list<Layer*>::iterator it = Layers.begin(); it != Layers.end(); it++)
		if ((*it)->GetId() == id)
			return (*it);
	return nullptr;
}

void Model::CopyFromModel(const Model &model)
{
	std::map<unsigned int, unsigned int> idMap;

	//copy Layers
	for (unsigned int i = 0; i < model.GetLayerCount(); ++i)
	{
		Layer* original = model.GetLayerAt(i);
		Layer* copy = original->Clone();
		AddLayer(copy, true);
		idMap.insert(std::pair<unsigned int, unsigned int>(original->GetId(), copy->GetId()));
	}

	//find parents
	for (unsigned int i = 0; i < GetLayerCount(); ++i)
	{
		Layer* findLayer = GetLayerAt(i);
		if (findLayer->GetInputLayer() == nullptr)
			continue;
		unsigned int idToFind = findLayer->GetInputLayer()->GetId();
		idToFind = idMap[idToFind];
		for (unsigned int j = 0; j < GetLayerCount(); ++j)
		{
			Layer* currentLayer = GetLayerAt(j);
			if (currentLayer->GetId() == idToFind)
			{
				findLayer->SetInput(currentLayer);
				break;
			}
		}
	}
}

void Model::Train(Optimizer *optimizer)
{
	if (inputLayer == nullptr || outputLayer == nullptr)
		throw LayerTypeException();
}

void Model::UpdateInputOutput()
{
	FindInput();
	FindOutput();
}

rapidjson::Document Model::SaveToDocument() const
{
	rapidjson::Document document;
	rapidjson::Value model(rapidjson::kObjectType);
	document.SetObject();

	rapidjson::Value layerList;
	layerList.SetArray();
	std::list<Layer*>::const_iterator iterator;

	for (iterator = Layers.begin(); iterator != Layers.end(); iterator++)
	{
		rapidjson::Value layerOut(rapidjson::kObjectType);
		rapidjson::Value layerValue = (*iterator)->SaveToJSONObject(document);
		layerOut.AddMember("layer", layerValue, document.GetAllocator());
		layerList.PushBack(layerOut, document.GetAllocator());
	}

	model.AddMember("layers", layerList, document.GetAllocator());
	document.AddMember("model", model, document.GetAllocator());

	return document;
}
