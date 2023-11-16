#pragma once
#include "NeuralNetwork/Layers/Layer.h"
#include "NeuralNetwork/ActivationFunctions.hpp"
#include <queue>
#include <memory>
#include "NeuralNetwork/Optimizers/Optimizer.h"

/**
 * @brief This layer uses a single weight matrix to calculate with past information.
*/
class RecurrentLayer : public Layer
{
public:
	/**
	 * @brief Creates a recurrent layer with the specified parameters.
	 * @param inputLayer The layer where the input is coming from.
	 * @param size The size of the layer.
	 * @param timeSteps How many steps will be taken in the BPTT.
	*/
	RecurrentLayer(Layer* inputLayer, unsigned int size, unsigned int timeSteps = 3);
	virtual ~RecurrentLayer();

	/**
	 * @brief Creates a copy of the layer.
	 * @return A pointer poininting to the copy.
	*/
	virtual Layer* Clone();

	/**
	 * @brief Runs the calculation, and stores the result in the output matrix.
	*/
	virtual void Compute();

	/**
	 * @brief Returns the output matrix
	 * @return Pointer to the output matrix
	*/
	virtual Tensor& GetOutput();

	/**
	 * @brief Runs the calculation and return with the output matrix.
	 * @return A pointer of the output matrix.
	*/
	virtual Tensor& ComputeAndGetOutput();

	/**
	 * @brief Sets the activation function we want to use.
	 * @param func A pointer to the activation function.
	*/
	virtual void SetActivationFunction(ActivationFunction* func);

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
	 * @brief Used to tell the layer to store values for later training. Set to true if you want to train your layer.
	 * @param mode Sets the mode.
	*/
	virtual void SetTrainingMode(bool mode);

	/**
	 * @brief Returns the weights of the layer.
	 * @return A matrix pointer of the weights.
	*/
	Matrix& GetWeights();

	/**
	 * @brief Return the bias of the layer.
	 * @return A matrix pointer of the bias values.
	*/
	Matrix& GetBias();

	/**
	 * @brief Returns the recursive weights of the layer.
	 * @return A matrix pointer of the recursive weights.
	*/
	Matrix& GetRecurrentWeights();

	/**
	 * @brief Loads the layer from JSON
	 * @param jsonData rapidjsson value type, containing the data for the layer
	 */
	virtual void LoadFromJSON(rapidjson::Value& jsonData);

	/**
	 * @brief Saves the layer into a JSON value object
	 * @param document A reference for the top document object
	 * @return A rapidjson value type containing the layer
	 */
	virtual rapidjson::Value SaveToJSONObject(rapidjson::Document& document) const;
private:
	unsigned int TimeSteps;
	ActivationFunction* function;

	unsigned int CurrentStep;
	unsigned int Size;

	Matrix Weights;
	Matrix Bias;
	Matrix RecursiveWeight;
	Tensor InnerState;
	Matrix RecursiveState;
    Matrix InnerRow;

	Matrix WeightError;
	Matrix BiasError;
	Matrix RecursiveWeightError;

	Tensor IncomingValues;
};

