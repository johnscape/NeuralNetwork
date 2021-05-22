#pragma once
#include "Layer.h"
/**
 * @brief This layer is responsible for copying matrices into the model. Use this as the first layer of your network.
*/
class InputLayer :
	public Layer
{
public:
	/**
	 * @brief Creates a new input layer with a specified size.
	 * @param size The size of your input.
	*/
	InputLayer(unsigned int size);
	virtual ~InputLayer() {}

	/**
	 * @brief Creates a deep copy of your layer.
	 * @return A pointer poininting to the copy.
	*/
	virtual Layer* Clone();

	/**
	 * @brief Runs the calculation, and stores the result in the output matrix.
	*/
	virtual void Compute();

	/**
	 * @brief Runs the calculation and return with the output matrix.
	 * @return A pointer of the output matrix.
	*/
	virtual Matrix& ComputeAndGetOutput();

	/**
	 * @brief Sets a matrix as a constant input for the layer.
	 * @param input The input matrix.
	*/
	virtual void SetInput(const Matrix& input);

	/**
	 * @brief Calculates the error inside of the layer based on the last output, the input and the error.
	 * @param error The error of the next layer, used to calculate this layer's error.
	 * @param recursive If set to true, it will call its input layer with its own error. Used to train the model.
	*/
	virtual void GetBackwardPass(const Matrix& error, bool recursive = false);

	/**
	 * @brief Modifies the weights inside of the layer based on an optimizer algorithm.
	 * @param optimizer A pointer to the optimizer class.
	*/
	virtual void Train(Optimizer* optimizer) {}

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
	virtual std::string SaveToJSON(const char* fileName = nullptr);
private:
	unsigned int Size;
};

