#pragma once
#include "Layer.h"
#include "Matrix.h"
#include "ActivationFunctions.hpp"
#include <vector>
#include <deque>

/**
 * @brief This class implements an LSTM layer
*/
class LSTM :
    public Layer
{
public:
    /**
     * @brief Creates a layer with an LSTM cell
     * @param inputLayer The layer where the input comes from
     * @param cellStateSize The size of the cell
     * @param timeSteps The past steps stored for training.
    */
    LSTM(Layer* inputLayer, unsigned int cellStateSize, unsigned int timeSteps = 3);
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
    virtual Matrix* GetOutput();

    /**
     * @brief Runs the Compute method then returns with the output matrix
     * @return The pointer of the updated output matrix
    */
    virtual Matrix* ComputeAndGetOutput();
    
    /**
     * @brief Calculates the error inside of the layer based on the last output, the input and the error.
     * @param error The error of the next layer, used to calculate this layer's error.
     * @param recursive If set to true, it will call its input layer with its own error.
    */
    virtual void GetBackwardPass(Matrix* error, bool recursive = false);

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
     * @param weight The selected gate
     * @return Matrix pointer of the specified input weight
    */
    Matrix* GetWeight(unsigned char weight);

    /**
     * @brief Returns the recursive weight from a selected gate.
     * @param weight The selected gate
     * @return Matrix pointer of the specified recursive weight
    */
    Matrix* GetRecursiveWeight(unsigned char weight);

    /**
     * @brief Returns the bias from a selected gate.
     * @param weight The selected gate
     * @return Matrix pointer of the specified bias
    */
    Matrix* GetBias(unsigned char weight);

    //TODO: select weight by enum

    /**
     * @brief Loads the layer from JSON.
     * @param data The JSON data containing the layer's information or the JSON file's name.
     * @param isFile If you want to load the data from a file, set it true.
    */
    virtual void LoadFromJSON(const char* data, bool isFile = false);

    /**
     * @brief Saves the layer into a JSON string.
     * @param fileName If you want to save the string into a file, set the filename here.
     * @return The JSON string describing the layer.
    */
    virtual std::string SaveToJSON(const char* fileName = nullptr);

    enum Gate
    {
        FORGET = 0,
        INPUT = 1,
        UPDATE = 2, //INPUT MODULATION
        OUTPUT = 3
    };

private:
    std::vector<Matrix*> InputWeights;
    std::vector<Matrix*> RecursiveWeights;
    std::vector<Matrix*> Biases;
    std::vector<Matrix*> InputWeightOutputs;
    std::vector<Matrix*> RecursiveWeightOuputs;

    std::vector<Matrix*> InputWeightErrors;
    std::vector<Matrix*> RecursiveWeightErrors;
    std::vector<Matrix*> BiasErrors;

    std::deque<std::vector<Matrix*>> savedStates;
    std::deque<Matrix*> errors;

    Matrix* CellState;
    Matrix* InnerState;

    Matrix* cellTanh;
    Matrix* DeltaOut;

    unsigned int CellStateSize;
    unsigned int TimeSteps;

    ActivationFunction* Tanh;
    ActivationFunction* Sig;

    void UpdateWeightErrors(Matrix* gateIError, Matrix* gateRError, Matrix* inputTranspose, Matrix* dGate, Matrix* outputTranspose, int weight);
};

