#pragma once

#include "NeuralNetwork/Tensor.h"
#include "NeuralNetwork/Layers/LayerException.hpp"

class Optimizer;

/**
 * @brief The layer class is representing a layer in the network.
*/
class Layer
{
public:
	/**
	 * @brief Enum for layer classification
	 */
	enum class LayerType
	{
		INPUT, /**< Input layer */
		FORWARD, /**< Feedforward layer */
		RECURRENT, /**< Simple recurrent layer to learn from past data  */
		LSTM, /**< Long short-term memory */
		CONV /**< Convolutional layer */
	};

	/**
	 * @brief Creates an abstract layer.
	 * @param inputLayer The layer where the input comes from. Set to null for an input layer.
	*/
	Layer(Layer* inputLayer);

	/**
	 * @brief Creates an abstract layer, with no input.
	*/
	Layer();

	/**
	 * @brief Creates a deep copy from a specified layer.
	 * @return The copied layer.
	*/
	virtual Layer* Clone() = 0;

	/**
	 * @brief Creates a new layer from the provided inputs.
	 * @param type The type of the layer (input, feed forward, LSTM, etc).
	 * @param size The size of the layer
	 * @return A new layer based on the parameters
	*/
	static Layer* Create(LayerType type, std::vector<unsigned int> size, Layer* input = nullptr);
	virtual ~Layer();

	/**
	 * @brief Sets the new input of this layer.
	 * @param input The layer where the input will be coming from.
	*/
	virtual void SetInput(Layer* input);

	/**
	 * @brief Sets a matrix as a constant input for the layer.
	 * @param input The input matrix
	*/
	virtual void SetInput(const Tensor& input) {}

	/**
	 * @brief Sets a tensor as a constant input for the layer.
	 * @param input The input tensor
	 */
	virtual void SetInput(const Matrix& input);

	/**
	 * @brief Calculates the output of the layer.
	*/
	virtual void Compute() = 0;

	/**
	 * @brief Returns the output matrix
	 * @return Pointer to the output matrix
	*/
	virtual Tensor& GetOutput();

	/**
	 * @brief Returns the second dimension of the output
	 * @return The second dimension of the output
	*/
	virtual unsigned int OutputSize();

	/**
	 * @brief Runs the Compute method then returns with the output matrix
	 * @return The pointer of the updated output matrix
	*/
	virtual Tensor& ComputeAndGetOutput() = 0;

	/**
	 * @brief Returns the layer where the input values are coming from.
	 * @return nullptr if there is no input layer, the pointer of the input layer otherwise.
	*/
	virtual Layer* GetInputLayer() const;

	/**
	 * @brief Calculates the error inside of the layer based on the last output, the input and the error.
	 * @param error The error of the next layer, used to calculate this layer's error.
	 * @param recursive If set to true, it will call its input layer with its own error.
	*/
	virtual void GetBackwardPass(const Tensor& error, bool recursive = false) = 0;

	/**
	 * @brief Modifies the weights inside of the layer based on an optimizer algorithm.
	 * @param optimizer A pointer to the optimizer class.
	*/
	virtual void Train(Optimizer* optimizer) = 0;

	/**
	 * @brief Returns with the error of the layer.
	 * @return The tensor where the layer's error is stored.
	*/
	virtual Tensor& GetLayerError();

	/**
	 * @brief Used to tell the layer to store values for later training. Set to true if you want to train your layer.
	 * @param mode Sets the mode.
	 * @param recursive If set to true, it will call the input layer with the same information. Used to set the whole model.
	*/
	virtual void SetTrainingMode(bool mode, bool recursive = true);

	/**
	 * @brief Creates a specified layer from JSON data.
	 * @param data The JSON data where the information is stored.
	 * @param isFile If the JSON data is stored in a file, you can read the file's content by setting it true, and passing the file's name at the data.
	 * @return A layer based on the JSON data.
	*/
	virtual Layer* CreateFromJSON(const char* data, bool isFile = false);

	/**
	 * @brief Loads the layer from JSON.
	 * @param data The JSON data containing the layer's information or the JSON file's name.
	 * @param isFile If you want to load the data from a file, set it true.
	*/
	virtual void LoadFromJSON(const char* data, bool isFile = false);

	/**
	 * @brief Loads the layer from JSON
	 * @param jsonData rapidjsson value type, containing the data for the layer
	 */
	virtual void LoadFromJSON(rapidjson::Value& jsonData) = 0;

	/**
	 * @brief Saves the layer into a JSON string.
	 * @param fileName If you want to save the string into a file, set the filename here.
	 * @return The JSON string describing the layer.
	*/
	virtual std::string SaveToJSON(const char* fileName = nullptr) const;

	/**
	 * @brief Saves the layer into a JSON value object
	 * @param document A reference for the top document object
	 * @return A rapidjson value type containing the layer
	 */
	virtual rapidjson::Value SaveToJSONObject(rapidjson::Document& document) const = 0;

	/**
	 * @brief Returns the layer unique id.
	 * @return Unsigned int, the layer's id.
	*/
	unsigned int GetId();

	/**
	 * @brief Sets the layer's id. If you do not know what it does, DO NOT USE!
	 * @param id The new id
	*/
	void SetId(unsigned int id);

    /**
     * Resets a layer's internal state, if it has one
     */
    virtual void Reset();

protected:
	Layer* LayerInput;
	Tensor Output;
	Tensor LayerError;

	unsigned int Id;
	bool TrainingMode;

	static unsigned int LayerCount;
};

