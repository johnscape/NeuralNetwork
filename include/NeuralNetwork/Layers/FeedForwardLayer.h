#pragma once
#include "NeuralNetwork/Layers/Layer.h"
#include "NeuralNetwork/ActivationFunctions.hpp"

/**
 * @brief This class represents a simple feed forward layer.
*/
class FeedForwardLayer :
	public Layer
{
public:
	/**
	 * @brief Creates a new feed forward layer.
	 * @param inputLayer The layer where the input comes from.
	 * @param count The size of the layer.
	*/
	FeedForwardLayer(Layer* inputLayer, unsigned int count);

	/**
	 * @brief Creates a copy of the layer.
	 * @return A pointer poininting to the copy.
	*/
	virtual Layer* Clone();
	virtual ~FeedForwardLayer();

	/**
	 * @brief Sets the layer where the input will come from.
	 * @param input A pointer to the input layer.
	*/
	virtual void SetInput(Layer* input);

	/**
	 * @brief Runs the calculation, and stores the result in the output matrix.
	*/
	virtual void Compute();

	/**
	 * @brief Runs the calculation and return with the output matrix.
	 * @return A pointer of the output matrix.
	*/
	virtual Tensor& ComputeAndGetOutput();

	/**
	 * @brief Sets the activation function we want to use.
	 * @param func A pointer to the activation function.
	*/
	void SetActivationFunction(ActivationFunction* func);

	/**
	 * @brief Calculates the error inside of the layer based on the last output, the input and the error.
	 * @param error The error of the next layer, used to calculate this layer's error.
	 * @param recursive If set to true, it will call its input layer with its own error. Used to train the model.
	*/
	virtual void GetBackwardPass(const Tensor& error, bool recursive = false);

	/**
	 * @brief Modifies the weights inside of the layer based on an optimizer algorithm.
	 * @param optimizer A pointer to the optimizer class.
	*/
	virtual void Train(Optimizer* optimizer);

	/**
	 * @brief Return the bias of the layer.
	 * @return A matrix pointer of the bias values.
	*/
	Matrix& GetBias();

	/**
	 * @brief Returns the weights of the layer.
	 * @return A matrix pointer of the weights.
	*/
	Matrix& GetWeights();

	/**
	 * @brief Loads the layer from a JSON string.
	 * @param data The JSON data or file name to load from.
	 * @param isFile If the data is from a file, set it to true.
	*/
	virtual void LoadFromJSON(const char* data, bool isFile = false);

	/**
	 * @brief Saves the layer into a JSON string.
	 * @param fileName If you want to save the JSON data into a file, enter the file name here.
	 * @return A string containing the JSON describtion of the layer.
	*/
	virtual std::string SaveToJSON(const char* fileName = nullptr) const;

private:

	ActivationFunction* function;


	Matrix Weights;
	Matrix Bias;
	Tensor InnerState;

	Matrix WeightError;
	Matrix BiasError;

	unsigned int Size;
};

