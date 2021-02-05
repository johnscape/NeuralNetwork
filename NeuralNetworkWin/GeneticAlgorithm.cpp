#include "GeneticAlgorithm.h"
#include "Layer.h"
#include <time.h>
#include "Model.h"

GeneticAlgorithm::GeneticAlgorithm(Layer* output, unsigned int generations, unsigned int individual_count) :
	Optimizer(output), MaxGenerations(generations), IndividialCount(individual_count), CurrentGeneration(0),
	MutationChance(0.2f), MutationMaxValue(0.5f), MaxParentCount(10)
{
	Layer* currentLayer = output;
	originalModel = new Model();
	while (currentLayer->GetInputLayer())
	{
		//originalLayers.push_back(currentLayer);
		originalModel->InsertFirstLayer(currentLayer);
		currentLayer = currentLayer->GetInputLayer();
	}

	srand(time(0));
}

GeneticAlgorithm::GeneticAlgorithm(Model* model, unsigned int generations, unsigned int individual_count) :
	Optimizer(model->GetOutput()), MaxGenerations(generations), IndividialCount(individual_count), CurrentGeneration(0),
	MutationChance(0.2f), MutationMaxValue(0.5f), MaxParentCount(10)
{
	originalModel = new Model(*model);
	srand(time(0));
}

float GeneticAlgorithm::MutationGenerator()
{
	if ((float)(rand() % 100) > MutationChance * 100.0f)
		return 0;
	unsigned int r = rand() % ((int)(MutationMaxValue * 1000) - (int)(-MutationMaxValue * 1000) + 1) + (int)(-MutationMaxValue * 1000);
	return (float)r / 1000.0f;
}

void GeneticAlgorithm::Mutate(Individual& individual)
{
	/*for (size_t i = 0; i < individual->layers.size(); i++)
		individual->layers[i]->Train(this);*/
	for (unsigned int i = 0; i < individual.model.LayerCount(); i++)
		individual.model.GetLayerAt(i)->Train(this);
}

Individual GeneticAlgorithm::CrossOver()
{
	Individual ind;
	/*for (size_t i = 0; i < originalLayers.size(); i++)
		ind->layers.push_back(parents[rand() % parents.size()]->layers[i]->Clone());*/
	for (unsigned int i = 0; i < originalModel->LayerCount(); i++)
	{
		ind.model.AddLayer(parents[rand() % parents.size()].model.GetLayerAt(i)->Clone());
		if (i > 0)
			ind.model.GetLayerAt(i)->SetInput(ind.model.GetLayerAt(i - 1));
	}
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
			Individual current;
			current.model = originalModel;
			Mutate(current);
			entities.push_back(current);
		}
	}
	else
	{
		for (size_t i = 0; i < IndividialCount - parents.size(); i++)
		{
			Individual current = CrossOver();
			Mutate(current);
			entities.push_back(current);
		}
	}
}

void GeneticAlgorithm::DeleteIndividuals()
{
	entities.clear();
}

void GeneticAlgorithm::Train(Matrix* input, Matrix* expected)
{
	if (parents.size() > 0)
		parents.clear();
	std::vector<unsigned int> coords;
	for (unsigned int p = 0; p < MaxParentCount; p++)
	{
		float maxVal = 0;
		unsigned int maxValPos = 0;
		for (unsigned int i = 0; i < entities.size(); i++)
		{
			if (entities[i].fitness > maxVal)
			{
				maxVal = entities[i].fitness;
				maxValPos = i;
			}
		}

		coords.push_back(maxValPos);
		entities[maxValPos].fitness = 0;
	}

	for (size_t i = 0; i < coords.size(); i++)
	{
		Individual p;
		p.fitness = entities[coords[i]].fitness;
		//p.layers = GetCopy(entities[coords[i]]->layers);
		p.model = entities[coords[i]].model;
		parents.push_back(p);
	}
}

void GeneticAlgorithm::ModifyWeights(Matrix* weights, Matrix* errors)
{
	for (size_t r = 0; r < weights->GetRowCount() * weights->GetColumnCount(); r++)
		weights->AdjustValue(r, MutationGenerator());
}

Individual& GeneticAlgorithm::GetIndividual(unsigned int num)
{
	if (num < 0 || num >= IndividialCount)
	{
		Individual i;
		return i;
	}
	return entities[num];
}
