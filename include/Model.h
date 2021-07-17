#pragma once

#include <vector>

class Layer;
class Matrix;

/**
 * @brief Used to store multiple layers belonging to one model.
*/
class Model
{
public:
	/**
	 * @brief Creates a new model
	*/
	Model();
	~Model();

	/**
	 * @brief Adds a layer to the model
	 * @param layer The layer
	*/
	void AddLayer(Layer* layer);

	/**
	 * @brief Returns a layer with a specified id
	 * @param id The id
	 * @return The layer with the id
	*/
	Layer* GetLayer(unsigned int id);

	/**
	 * @brief Saves the model into a JSON file
	 * @param fileName The JSON file
	*/
	void SaveModel(const char* fileName);

	/**
	 * @brief Loads the model from a JSON file
	 * @param fileName The JSON file
	*/
	void LoadModel(const char* fileName);

	/**
	 * @brief Computes the model
	 * @param input The input of the model
	 * @return The output of the final layer
	*/
	Matrix* Compute(Matrix* input);

	/**
	 * @brief Returns the last added layer
	 * @return The layer
	*/
	Layer* GetLastLayer();

	/**
	 * @brief Returns the output layer of the model
	 * @return The output layer
	*/
	Layer* GetOutput();

	/**
	 * @brief Returns the input layer of the model
	 * @return The input layer
	*/
	Layer* GetInput();

private:
	std::vector<Layer*> layers;
	Layer* outputLayer;
	Layer* inputLayer;

	void FindOutput();
	void FindInput();
	Layer* FindLayerWithId(unsigned int id);
};

