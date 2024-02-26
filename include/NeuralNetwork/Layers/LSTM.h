#pragma once
#include "NeuralNetwork/Matrix.h"
#include "NeuralNetwork/Tensor.h"
#include "NeuralNetwork/ActivationFunctions.hpp"
#include "NeuralNetwork/Layers/Layer.h"
#include <vector>
#include <deque>

/**
 * @brief This class implements an LSTM layer
*/
class LSTM :
    public Layer
{
public:
	enum class Gate
	{
		FORGET,
		INPUT,
		ACTIVATION,
		OUTPUT
	};

    /**
     * @brief Creates a layer with an LSTM cell
     * @param inputLayer The layer where the input comes from
     * @param cellStateSize The size of the cell
    */
    LSTM(Layer* inputLayer, unsigned int cellStateSize);
    virtual ~LSTM();

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
     * @brief Runs the Compute method then returns with the output matrix
     * @return The pointer of the updated output matrix
    */
    virtual Tensor& ComputeAndGetOutput();
    
    /**
     * @brief Calculates the error inside of the layer based on the last output, the input and the error.
     * @param error The error of the next layer, used to calculate this layer's error.
     * @param recursive If set to true, it will call its input layer with its own error.
    */
    virtual void GetBackwardPass(const Tensor& error, bool recursive = false);

    /**
     * @brief Modifies the weights inside of the layer based on an optimizer algorithm.
     * @param optimizer A pointer to the optimizer class.
    */
    virtual void Train(Optimizer* optimizer);

    /**
     * @brief Tells the layer to store values for later training. Set to true if you want to train your layer.
     * @param mode Sets the mode.
     * @param recursive If set to true, it will call the input layer with the same information. Used to set the whole model.
    */
    virtual void SetTrainingMode(bool mode, bool recursive = false);

	/**
	 * @brief Returns the input weight from a selected gate.
	 * @param gate The selected gate
	 * @return Matrix pointer of the specified input weight
	 */
	Matrix& GetWeight(Gate gate);

	/**
	 * @brief Returns the bias from a selected gate.
	 * @param gate The selected gate
	 * @return Matrix pointer of the specified bias
	 */
	Matrix& GetBias(Gate gate);

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
    Matrix ForgetGateWeight;
    Matrix ForgetGateBias;

    Matrix InputGateSigmoidWeight;
    Matrix InputGateSigmoidBias;
    Matrix InputGateTanhWeight;
    Matrix InputGateTanhBias;

    Matrix OutputGateWeight;
    Matrix OutputGateBias;

    Matrix CellState;
    Matrix InnerState;

    unsigned int CellStateSize;
    unsigned int InputSize;

    Matrix ForgetGateWeightError;
    Matrix ForgetGateBiasError;

    Matrix InputGateSigmoidWeightError;
    Matrix InputGateSigmoidBiasError;
    Matrix InputGateTanhWeightError;
    Matrix InputGateTanhBiasError;

    Matrix OutputGateWeightError;
    Matrix OutputGateBiasError;


    ActivationFunction* Tanh;
    ActivationFunction* Sig;

    void UpdateWeightErrors(Matrix& gateIError, Matrix& gateRError, Matrix& inputTranspose, Matrix& dGate, Matrix& outputTranspose, int weight);
};

