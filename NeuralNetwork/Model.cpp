#include "Model.h"

#include "Layer.h"
#include "Matrix.h"

Model::Model() : inputLayer(nullptr), outputLayer(nullptr)
{
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

Layer* Model::GetLayer(unsigned int id)
{
	if (id >= 0 && id < layers.size() - 1)
		return layers[id];
	return nullptr;
}

void Model::SaveModel(const char* fileName)
{
}

void Model::LoadModel(const char* fileName)
{
}

Matrix Model::Compute(Matrix* input)
{
	return Matrix();
}

void Model::FindOutput()
{
	bool found = true;
	for (unsigned int i = 0; i < layers.size(); i++)
	{
		for (unsigned int ii = 0; ii < layers.size(); ii++)
		{
			if (i == ii)
				continue;
			if (layers[ii]->GetInputLayer() == layers[i])
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
