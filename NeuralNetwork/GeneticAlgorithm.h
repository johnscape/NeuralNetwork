#pragma once
#include "Optimizer.h"
#include <vector>

typedef float (*Fitness)(Matrix*);
class Layer;

struct Individual
{
    std::vector<Layer*> layers;
    float fitness;

    ~Individual()
    {
        for (size_t i = 0; i < layers.size(); i++)
            delete layers[i];
    }
};

class GeneticAlgorithm :
    public Optimizer
{
public:
    GeneticAlgorithm(Layer* output, unsigned int generations, unsigned int individual_count);
    ~GeneticAlgorithm();

    void Mutate(Individual* individual);
    Individual* CrossOver();

    void GenerateIndividuals();
    void DeleteIndividuals();

    virtual void Train(Matrix* input, Matrix* expected);
    virtual void ModifyWeights(Matrix* weights, Matrix* errors);
    Individual* GetIndividual(unsigned int num);

private:
    float MutationChance;
    float MutationMaxValue;
    unsigned int MaxParentCount;
    unsigned int MaxGenerations;
    unsigned int CurrentGeneration;
    unsigned int IndividialCount;
    std::vector<Layer*> originalLayers;

    std::vector<Individual*> entities;
    std::vector<Individual*> parents;

    std::vector<Layer*> GetCopy(std::vector<Layer*>& layers);
    float MutationGenerator();
};

