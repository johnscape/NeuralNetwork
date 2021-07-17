#include "NeuralNetwork/Model.h"

#include "NeuralNetwork/Layer.h"
#include "NeuralNetwork/Matrix.h"
#include <map>
#include "NeuralNetwork/Constants.h"

#include "rapidjson/document.h"
#include <fstream>
#include <string>

Model::Model() : inputLayer(nullptr), outputLayer(nullptr)
{
}

Model::Model(const Model& m)
{
	Layer* currentLayer = m.outputLayer->Clone();
	AddLayer(currentLayer);
	while (currentLayer->GetInputLayer())
	{
		currentLayer = currentLayer->GetInputLayer()->Clone();
		layers[0]->SetInput(currentLayer);
		InsertFirstLayer(currentLayer);
	}
}

Model& Model::operator=(Model& other)
{
	if (layers.size() > 0)
	{
		for (unsigned int i = 0; i < layers.size(); i++)
			delete layers[i];
		layers.clear();
	}

	Layer* currentLayer = other.outputLayer->Clone();
	AddLayer(currentLayer);
	while (currentLayer->GetInputLayer())
	{
		currentLayer = currentLayer->GetInputLayer()->Clone();
		layers[0]->SetInput(currentLayer);
		InsertFirstLayer(currentLayer);
	}

	return *this;
}

Model& Model::operator=(Model* other)
{
	if (layers.size() > 0)
	{
		for (unsigned int i = 0; i < layers.size(); i++)
			delete layers[i];
		layers.clear();
	}

	Layer* currentLayer = other->outputLayer->Clone();
	AddLayer(currentLayer);
	while (currentLayer->GetInputLayer())
	{
		currentLayer = currentLayer->GetInputLayer()->Clone();
		layers[0]->SetInput(currentLayer);
		InsertFirstLayer(currentLayer);
	}

	return *this;
}

Model::~Model()
{
	for (unsigned int i = 0; i < layers.size(); i++)
		delete layers[i];
}

void Model::AddLayer(Layer* layer)
{
	layers.push_back(layer);
	if (layers.size() >= 2)
		layers[layers.size() - 1]->SetInput(layers[layers.size() - 2]);
	if (!inputLayer)
		inputLayer = layer;
	FindOutput();
}

void Model::InsertFirstLayer(Layer* layer)
{
	layers.insert(layers.begin(), layer);
	inputLayer = layer;
	FindOutput();
}

Layer* Model::GetLayer(unsigned int id)
{
	return FindLayerWithId(id);
}

void Model::SaveModel(const char* fileName)
{
	/*rapidjson::Document document;
	document.SetObject();

	rapidjson::Value modelData(rapidjson::kObjectType);
	rapidjson::Value layerList(rapidjson::kArrayType);
	rapidjson::Value inputId, layerCount, inpLayer, outLayer;
	rapidjson::Document layerInfo;
	for (unsigned int i = 0; i < layers.size(); i++)
	{
		rapidjson::Value layerData(rapidjson::kObjectType);
		if (layers[i]->GetInputLayer())
			inputId.SetInt(layers[i]->GetInputLayer()->GetId());
		else
			inputId.SetInt(-1);
		layerInfo.Parse(layers[i]->SaveToJSON().c_str());
		layerData.AddMember("inputLayerId", inputId, document.GetAllocator());
		layerData.AddMember("layerData", layerInfo, document.GetAllocator());
		layerList.PushBack(layerData, document.GetAllocator());
	}

	layerCount.SetUint(layers.size());
	inpLayer.SetUint(inputLayer->GetId());
	outLayer.SetUint(outputLayer->GetId());

	modelData.AddMember("layers", layerList, document.GetAllocator());
	modelData.AddMember("layerCount", layerCount, document.GetAllocator());
	modelData.AddMember("inputLayerId", inpLayer, document.GetAllocator());
	modelData.AddMember("outputLayerId", outLayer, document.GetAllocator());
	document.AddMember("model", modelData, document.GetAllocator());

	std::ofstream w(fileName);
	rapidjson::OStreamWrapper osw(w);
	rapidjson::Writer<rapidjson::OStreamWrapper> writer(osw);
	document.Accept(writer);
	w.close();*/
}

void Model::LoadModel(const char* fileName)
{
	/*rapidjson::Document document;
	std::ifstream r(fileName);
	rapidjson::IStreamWrapper isw(r);
	document.ParseStream(isw);

	if (layers.size() > 0)
	{
		for (unsigned int i = 0; i < layers.size(); i++)
			delete layers[i];
		layers.clear();
	}

	rapidjson::Value layerList, layerCount;
	layerList = document["model"]["layers"];
	layerCount = document["model"]["layerCount"];

	rapidjson::StringBuffer buffer;
	rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);

	std::map<int, int> layerPairs;

	for (unsigned int i = 0; i < layerCount.GetUint(); i++)
	{
		Layer* input = FindLayerWithId(layerList[i]["inputLayerId"].GetInt());
		Layer* layer = Layer::Create(layerList[i]["layerData"]["layer"]["type"].GetUint(),
			layerList[i]["layerData"]["layer"]["size"].GetUint(), input);

		layerList[i]["layerData"].Accept(writer);
		std::string layerString = buffer.GetString();
		layer->LoadFromJSON(layerString.c_str());

		buffer.Clear();
		writer.Reset(buffer);

		layerPairs.insert(std::make_pair<int, int>(layerList[i]["layerData"]["layer"]["id"].GetInt(), layerList[i]["inputLayerId"].GetInt()));
		layer->SetId(layerList[i]["layerData"]["layer"]["id"].GetUint());
		layers.push_back(layer);
	}

	for (std::map<int, int>::iterator it = layerPairs.begin(); it != layerPairs.end(); it++)
	{
		if (it->second < 0)
			continue;
		Layer* currentLayer = FindLayerWithId(it->first);
		if (currentLayer->GetInputLayer())
			continue;
		currentLayer->SetInput(FindLayerWithId(it->second));
	}

	inputLayer = FindLayerWithId(document["model"]["inputLayerId"].GetUint());
	outputLayer = FindLayerWithId(document["model"]["outputLayerId"].GetUint());*/
}

Matrix Model::Compute(const Matrix& input)
{
	inputLayer->SetInput(input);
	outputLayer->Compute();
#if USE_GPU
	outputLayer->GetOutput()->CopyFromGPU();
#endif // USE_GPU

	return outputLayer->GetOutput();
}

Layer* Model::GetLastLayer()
{
	if (layers.size() > 0)
		return layers[layers.size() - 1];
	return nullptr;
}

Layer* Model::GetOutput()
{
	return outputLayer;
}

Layer* Model::GetInput()
{
	return inputLayer;
}

unsigned int Model::LayerCount() const
{
	return layers.size();
}

Layer* Model::GetLayerAt(unsigned int n)
{
	if (n >= layers.size())
		return nullptr;
	return layers[n];
}

void Model::FindOutput()
{
	for (unsigned int i = 0; i < layers.size(); i++)
	{
		bool found = true;
		for (unsigned int ii = 0; ii < layers.size(); ii++)
		{
			if (i == ii)
				continue;
			if (layers[ii]->GetInputLayer() && layers[ii]->GetInputLayer()->GetId() == layers[i]->GetId())
			{
				found = false;
				break;
			}
		}

		if (found)
		{
			outputLayer = layers[i];
			break;
		}
	}
}

void Model::FindInput()
{
	for (unsigned int i = 0; i < layers.size(); i++)
	{
		if (!layers[i]->GetInputLayer())
		{
			inputLayer = layers[i];
			return;
		}
	}
}

Layer* Model::FindLayerWithId(unsigned int id)
{
	for (unsigned int i = 0; i < layers.size(); i++)
		if (layers[i]->GetId() == id)
			return layers[i];
	return nullptr;
}
