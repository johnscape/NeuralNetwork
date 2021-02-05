#pragma once
#include "Optimizer.h"
#include <vector>
#include "Model.h"

typedef float (*Fitness)(Matrix*);
class Layer;

/**
 * @brief Represents an individual for the genetic algorithm
*/
struct Individual //TODO: Use model instead vector
{
    Model model;
    float fitness;

    Individual() : model()
    {
        fitness = 0;
    }
};

/**
 * @brief Genetic algorithm to train the network unsupervised.
*/
class GeneticAlgorithm :
    public Optimizer
{
public:
    /**
     * @brief Creates a genetic algorithm optimizer.
     * @param output The output of the network.
     * @param generations The number of generations to train.
     * @param individual_count The number of individuals in a generation.
    */
    GeneticAlgorithm(Layer* output, unsigned int generations, unsigned int individual_count);
    GeneticAlgorithm(Model* model, unsigned int generations, unsigned int individual_count);

    /**
     * @brief Mutates an individual based on the mutation settings.
     * @param individual The individual to mutate.
    */
    void Mutate(Individual& individual);

    /**
     * @brief Creates a new individual from the values of the parents.
     * @return A new individual.
    */
    Individual CrossOver();

    /**
     * @brief Creates a new generation of individuals.
    */
    void GenerateIndividuals();

    /**
     * @brief Removes all individuals from the current generation.
    */
    void DeleteIndividuals();

    /**
     * @brief Trains the model based on the input and the expected output.
     * @param input The input of the model.
     * @param expected The expected output of the model.
    */
    virtual void Train(Matrix* input, Matrix* expected);

    /**
     * @brief Based on the type of the optimizer, this function will modify the weights of the layers.
     * @param weights The weights to modify
     * @param errors The error to calculate the new weights from
    */
    virtual void ModifyWeights(Matrix* weights, Matrix* errors);

    /**
     * @brief Returns an individual at a specific index.
     * @param num The individual's index.
     * @return The specified individual.
    */
    Individual& GetIndividual(unsigned int num);

private:
    float MutationChance;
    float MutationMaxValue;
    unsigned int MaxParentCount;
    unsigned int MaxGenerations;
    unsigned int CurrentGeneration;
    unsigned int IndividialCount;
    Model* originalModel;

    std::vector<Individual> entities;
    std::vector<Individual> parents;

    float MutationGenerator();
};

