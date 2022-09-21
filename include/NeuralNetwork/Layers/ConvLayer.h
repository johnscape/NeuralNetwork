#pragma once

#include "Layer.h"

class ActivationFunction;

/**
 * @brief A convolutional layer
 */
class ConvLayer : public Layer
{
public:
	/**
	 * @brief Creates a new convolutional layer
	 * @param inputLayer The layer to get the input from
	 * @param kernelSize The size of the kernel: nxn
	 * @param stride The stride, i.e.: the size of the steps the kernel makes
	 * @param nrFilters The number of kernels used, determines the size of the output
	 * @param minimumPad The minimum padding applied to the input
	 * @param padType The type of padding applied
	 * @param padFill If constant padding is used, this will be the value used to fill the pads
	 */
	ConvLayer(Layer* inputLayer, unsigned int kernelSize, unsigned int stride=1, unsigned int nrFilters = 1,
			  unsigned int minimumPad = 0, Matrix::PadType padType = Matrix::PadType::CONSTANT, float padFill = 0);
	virtual ~ConvLayer();

	/**
	 * @brief Creates a clone of this layer
	 * @return The clone of this layer
	 */
	virtual Layer* Clone();

	/**
	 * @brief Computes the output of the layer
	 */
	virtual void Compute();

	/**
	 * @brief Computes the output and returns it
	 * @return The updated output of the layer
	 */
	virtual Tensor& ComputeAndGetOutput();

	/**
	 * @brief Updates the layer's error for training purposes
	 * @param error The error of the next layer
	 * @param recursive Set to true if you want to train all previous Layers
	 */
	virtual void GetBackwardPass(const Tensor& error, bool recursive = false);

	/**
	 * @brief Uses an optimizer to update the kernel based on the error
	 * @param optimizer The optimizer to use
	 */
	virtual void Train(Optimizer* optimizer);

	/**
	 * @brief Returns the kernel of the layer
	 * @return The kernel of the layer
	 */
	Tensor& GetKernel();

	/**
	 * @brief Returns the pad size used in the convolution
	 * @return The size of the pad
	 */
	unsigned int GetPadSize();

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
	Tensor Kernel;
	Tensor InputStep; //Saved input for training purposes
	unsigned int PadSize;
	unsigned int Stride;
	Matrix::PadType PaddingType;
	float PadFill;

	Tensor KernelError;

	ActivationFunction* function;
};
