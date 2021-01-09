#include "GeneticAlgorithm.h"
#include "Layer.h"
#include <time.h>

GeneticAlgorithm::GeneticAlgorithm(Layer* output, unsigned int generations, unsigned int individual_count) :
	Optimizer(output), MaxGenerations(generations), IndividialCount(individual_count), CurrentGeneration(0),
	MutationChance(0.2f), MutationMaxValue(0.5f), MaxParentCount(10)
{
	Layer* currentLayer = output;
	while (currentLayer->GetInputLayer())
	{
		originalLayers.push_back(currentLayer);
		currentLayer = currentLayer->GetInputLayer();
	}
	srand(time(0));
}

GeneticAlgorithm::~GeneticAlgorithm()
{
	for (size_t i = 0; i < entities.size(); i++)
		delete entities[i];

	for (size_t i = 0; i < parents.size(); i++)
		delete parents[i];
}

std::vector<Layer*> GeneticAlgorithm::GetCopy(std::vector<Layer*>& layers)
{
	std::vector<Layer*> newLayers;
	for (unsigned int i = 0; i < layers.size(); i++)
	{
		newLayers.push_back(layers[i]->Clone());
		if (i)
			newLayers[i - 1]->SetInput(newLayers[i]);
	}
	return newLayers;
}

float GeneticAlgorithm::MutationGenerator()
{
	if ((float)(rand() % 100) > MutationChance * 100.0f)
		return 0;
	unsigned int r = rand() % ((int)(MutationMaxValue * 1000) - (int)(-MutationMaxValue * 1000) + 1) + (int)(-MutationMaxValue * 1000);
	return (float)r / 1000.0f;
}

void GeneticAlgorithm::Mutate(Individual* individual)
{
	for (size_t i = 0; i < individual->layers.size(); i++)
		individual->layers[i]->Train(this);
}

Individual* GeneticAlgorithm::CrossOver()
{
	Individual* ind = new Individual();
	ind->fitness = 0;
	for (size_t i = 0; i < originalLayers.size(); i++)
		ind->layers.push_back(parents[rand() % parents.size()]->layers[i]->Clone());
	return ind;
}

void GeneticAlgorithm::GenerateIndividuals()
{
	if (entities.size() > 0)
		DeleteIndividuals();
	if (parents.size() == 0)
	{
		for (size_t i = 0; i < IndividialCount; i++)
		{
			Individual* current = new Individual();
			current->layers = GetCopy(originalLayers);
			Mutate(current);
			entities.push_back(current);
		}
	}
	else
	{
		for (size_t i = 0; i < IndividialCount - parents.size(); i++)
		{
			Individual* current = CrossOver();
			Mutate(current);
			entities.push_back(current);
		}
	}
}

void GeneticAlgorithm::DeleteIndividuals()
{
	for (size_t i = 0; i < entities.size(); i++)
		delete entities[i];
	entities.clear();
}

void GeneticAlgorithm::Train(Matrix* input, Matrix* expected)
{
	if (parents.size() > 0)
	{
		for (size_t i = 0; i < parents.size(); i++)
			delete parents[i];
		parents.clear();
	}
	std::vector<unsigned int> coords;
	for (unsigned int p = 0; p < MaxParentCount; p++)
	{
		float maxVal = 0;
		unsigned int maxValPos = 0;
		for (unsigned int i = 0; i < entities.size(); i++)
		{
			if (entities[i]->fitness > maxVal)
			{
				maxVal = entities[i]->fitness;
				maxValPos = i;
			}
		}

		coords.push_back(maxValPos);
		entities[maxValPos]->fitness = 0;
	}

	for (size_t i = 0; i < coords.size(); i++)
	{
		Individual* p = new Individual();
		p->fitness = entities[coords[i]]->fitness;
		p->layers = GetCopy(entities[coords[i]]->layers);
		parents.push_back(p);
	}
}

void GeneticAlgorithm::ModifyWeights(Matrix* weights, Matrix* errors)
{
	for (size_t r = 0; r < weights->GetRowCount() * weights->GetColumnCount(); r++)
		weights->AdjustValue(r, MutationGenerator());
}

Individual* GeneticAlgorithm::GetIndividual(unsigned int num)
{
	if (num < 0 || num >= IndividialCount)
		return nullptr;
	return entities[num];
}
