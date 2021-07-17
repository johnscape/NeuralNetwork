#include "NeuralNetwork/GeneticAlgorithm.h"
#include "NeuralNetwork/Layer.h"
#include <time.h>
#include "NeuralNetwork/Model.h"
#include "NeuralNetwork/Constants.h"
#include <iostream>
GeneticAlgorithm::GeneticAlgorithm(Layer* output, unsigned int generations, unsigned int individual_count, Fitness fitness) :
	Optimizer(output), MaxGenerations(generations), IndividialCount(individual_count), CurrentGeneration(0),
	MutationChance(0.5f), MutationMaxValue(0.01f), MaxParentCount(3), fitnessFunc(fitness)
{
	Layer* currentLayer = output;
	originalModel = new Model();
	while (currentLayer->GetInputLayer())
	{
		//originalLayers.push_back(currentLayer);
		originalModel->InsertFirstLayer(currentLayer);
		currentLayer = currentLayer->GetInputLayer();
	}
	InitializeRandom();
}

GeneticAlgorithm::GeneticAlgorithm(Model* model, unsigned int generations, unsigned int individual_count, Fitness fitness) :
	Optimizer(model->GetOutput()), MaxGenerations(generations), IndividialCount(individual_count), CurrentGeneration(0),
	MutationChance(0.5f), MutationMaxValue(0.01f), MaxParentCount(3), fitnessFunc(fitness)
{
	originalModel = new Model(*model);
	InitializeRandom();
}

float GeneticAlgorithm::MutationGenerator()
{
	/*float g = rand() % 100;
	g -= (1 - MutationChance) * 100.0f;
	if (g < 0)
		return 0;*/
	/*signed int r = rand() % ((int)(MutationMaxValue * 10000) - (int)(-MutationMaxValue * 10000) + 1) + (int)(-MutationMaxValue * 10000);
	float m = (float)r / 10000.0f;*/

	float random = ((float) rand() / (RAND_MAX)) - 0.5f; //TODO: Apply MutationMaxValue

	return random;
}

void GeneticAlgorithm::DoGeneration()
{
	GenerateIndividuals();
	//float sum = 0;
	//TODO: Add option to print current status
	for (unsigned int i = 0; i < entities.size(); i++)
	{
		entities[i].fitness = fitnessFunc(&entities[i].model);
	}

	FindParents();

}

void GeneticAlgorithm::FindParents()
{
	parents.clear();
	for (unsigned int i = 0; i < MaxParentCount; i++)
	{
		
		bool high = true; //TODO: Create property from it
		//find highest fitness
		unsigned int maxPoint = 0;
		float maxFittness = 0;
		if (!high)
			maxFittness = 100;
		for (size_t ii = 0; ii < entities.size(); ii++)
		{
			if (high && entities[ii].fitness > maxFittness)
			{
				maxFittness = entities[ii].fitness;
				maxPoint = ii;
			}
			else if (!high && entities[ii].fitness < maxFittness)
			{
				maxFittness = entities[ii].fitness;
				maxPoint = ii;
			}
		}
		parents.push_back(Individual(entities[maxPoint]));
	
		entities[maxPoint].fitness = high ? -10 : 100;


	}
}

void GeneticAlgorithm::InitializeRandom()
{
	/*std::random_device device;
	engine(device());
	distribution({})*/
	srand(time(0)); //TODO: Use distribution
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
	
	Individual ind; //(parents[rand() % parents.size()])

	for (unsigned int l = 0; l < originalModel->LayerCount(); l++)
	{
		unsigned int parent = rand() % parents.size();
		ind.model.AddLayer(parents[parent].model.GetLayerAt(l)->Clone());
		if (l > 0)
			ind.model.GetLayerAt(l)->SetInput(ind.model.GetLayerAt(l - 1));
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
			Individual current(*originalModel, 0);
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
		for (size_t i = 0; i < parents.size(); i++)
		{
			entities.push_back(Individual(parents[i]));
		}
	}
}

void GeneticAlgorithm::DeleteIndividuals()
{
	entities.clear();
}

void GeneticAlgorithm::Train(const Matrix& input, const Matrix& expected)
{
	CurrentGeneration = 0;
	for (size_t g = 0; g < MaxGenerations; g++)
		TrainStep(Matrix(), Matrix());
}

void GeneticAlgorithm::ModifyWeights(Matrix& weights, const Matrix& errors)
{
	for (size_t r = 0; r < weights.GetRowCount() * weights.GetColumnCount(); r++)
		weights.AdjustValue(r, MutationGenerator());
#if USE_GPU
	weights->CopyToGPU();
#endif // USE_GPU

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

void GeneticAlgorithm::SetFitnessFunc(Fitness fitness)
{
	fitnessFunc = fitness;
}

void GeneticAlgorithm::TrainStep(const Matrix& input, const Matrix& output)
{
	std::cout << "Current generation: " << CurrentGeneration << std::endl;
	DoGeneration();
	CurrentGeneration++;
}

void GeneticAlgorithm::Reset()
{
	CurrentGeneration = 0;
	entities.clear();
	parents.clear();
	originalModel = nullptr;
}
