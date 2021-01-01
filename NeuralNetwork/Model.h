#pragma once

#include <vector>

class Layer;
class Matrix;

class Model
{
public:
	Model();
	~Model();

	void AddLayer(Layer* layer);
	Layer* GetLayer(unsigned int id);

	void SaveModel(const char* fileName);
	void LoadModel(const char* fileName);

	Matrix Compute(Matrix* input);

private:
	std::vector<Layer*> layers;
	Layer* outputLayer;
	Layer* inputLayer;

	void FindOutput();
};

