#pragma once

#include "Matrix.h"
#include <memory>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "rapidjson/ostreamwrapper.h"
#include "rapidjson/istreamwrapper.h"
#include <fstream>

class Optimizer;

class Layer
{
public:
	Layer(Layer* inputLayer);
	Layer();
	virtual Layer* Clone() = 0;
	Layer* Create(unsigned int id, unsigned int size);
	virtual ~Layer();

	virtual void SetInput(Layer* input);
	virtual void SetInput(Matrix* input) {}

	virtual void Compute() = 0;
	virtual Matrix* GetOutput();
	virtual unsigned int OutputSize();
	virtual Matrix* ComputeAndGetOutput() = 0;

	virtual Layer* GetInputLayer();

	virtual void GetBackwardPass(Matrix* error, bool recursive = false) = 0;
	virtual void Train(Optimizer* optimizer) = 0;
	virtual Matrix* GetLayerError();

	virtual void SetTrainingMode(bool mode, bool recursive = true);

	virtual Layer* CreateFromJSON(const char* data, bool isFile = false);
	virtual void LoadFromJSON(const char* data, bool isFile = false) = 0;
	virtual std::string SaveToJSON(const char* fileName = nullptr) = 0;

	unsigned int GetId();

protected:
	Layer* LayerInput;
	Matrix* Output;
	Matrix* LayerError;

	static unsigned int Id;
	bool TrainingMode;
};

