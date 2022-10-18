#pragma once

#include <list>
#include <string>
#include "rapidjson/document.h"

class Layer;
class Matrix;
class Tensor;
class Optimizer;

/**
 * @brief Used to store multiple Layers belonging to one model.
*/
class Model
{
public:
	/**
	 * @brief Creates a new model
	*/
	Model();
	Model(const Model& m);
	Model& operator= (const Model& other);
	~Model();

	/**
	 * @brief Adds a layer to the model
	 * @param layer The layer to add
	 * @param toDelete Set to true, if you want to delete the layer in the deconstructor
	 */
	void AddLayer(Layer* layer, bool toDelete = false);

	/**
	 * @brief Returns a layer with a specified id
	 * @param id The id
	 * @return The layer with the id
	*/
	Layer* GetLayer(unsigned int id);

	/**
	 * @brief Gets the number of layers in the model
	 * @return The number of layers
	 */
	unsigned int GetLayerCount() const;

	/**
	 * @brief Saves the model into a JSON file
	 * @param fileName The JSON file
	*/
	void SaveModel(const char* fileName) const;

	/**
	 * @brief Loads the model from a JSON file
	 * @param fileName The JSON file
	*/
	void LoadModel(const char* fileName);

	/**
	 * @brief Saves the model into a JSON string
	 */
	std::string SaveToString() const;

	/**
	 * @brief Loads the model from a string
	 * @param json The model in a json string
	 */
	void LoadFromString(const std::string& json);

	/**
	 * @brief Loads the model from a string
	 * @param json The model in a json string
	 */
	void LoadFromString(const char* json);

	/**
	 * @brief Computes the model
	 * @param input The input of the model
	 * @return The output of the final layer
	*/
	Tensor Compute(const Tensor& input) const;

	/**
	 * @brief Returns the last added layer
	 * @return The layer
	*/
	Layer* GetLastLayer() const;

	/**
	 * @brief Returns the output layer of the model
	 * @return The output layer
	*/
	Layer* GetOutput() const;

	/**
	 * @brief Returns the input layer of the model
	 * @return The input layer
	*/
	Layer* GetInput() const;

	/**
	 * @brief Returns the number of the layers in the network
	 * @return The number of the layers
	*/
	unsigned int LayerCount() const;

	/**
	 * @brief Returns the nth layer in the model.
	 * @param pos Which layer to return
	 * @return The layer
	*/
	Layer* GetLayerAt(unsigned int n) const;

	/**
	 * @brief Trains the model with a given optimizer
	 * @param optimizer The optimizer to train with
	 */
	void Train(Optimizer* optimizer);

private:
	std::list<Layer*> Layers;
	std::list<bool> ToDelete;
	Layer* outputLayer;
	Layer* inputLayer;

	void FindOutput();
	void FindInput();
	Layer* FindLayerWithId(unsigned int id);
	void CopyFromModel(const Model& model);
	void UpdateInputOutput();

	rapidjson::Document SaveToDocument() const;
};