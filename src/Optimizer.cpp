#include "NeuralNetwork/Optimizer.h"

Optimizer::Optimizer(Layer* output) : currentBatch(1), lastError(0)
{
	outputLayer = output;
	//inputLayer = nullptr;
	FindInputLayer();
}

Optimizer::~Optimizer()
{
}

void Optimizer::TrainFor(const Tensor& input, const Tensor& expected, unsigned int times, unsigned int batch)
{
	FindInputLayer();
	SetTrainingMode(true);
	/*for (unsigned int time = 0; time < times; time++)
	{
		currentBatch = 0;
		for (unsigned int pos = 0; pos < input.GetRowCount(); pos++)
		{
			Tensor inp = input.GetRowMatrix(pos);
			Tensor exp = expected.GetRowMatrix(pos);
			currentBatch++;
			TrainStep(inp, exp);
			if (currentBatch >= batch)
			{
				TrainLayers();
				currentBatch = 0;
			}
		}
		if (currentBatch > 0)
			TrainLayers();
	}*/
	SetTrainingMode(false);
}

void Optimizer::TrainUntil(const Matrix& input, const Matrix& expected, float error, unsigned int batch)
{
}

void Optimizer::SetTrainingMode(bool mode)
{
	Layer* currentLayer = outputLayer;
	while (currentLayer->GetInputLayer())
	{
		currentLayer->SetTrainingMode(mode);
		currentLayer = currentLayer->GetInputLayer();
	}
}

void Optimizer::FindInputLayer()
{
	Layer* currentLayer = outputLayer;
	while (currentLayer->GetInputLayer())
		currentLayer = currentLayer->GetInputLayer();
	inputLayer = currentLayer;
}

void Optimizer::TrainLayers()
{
	Layer* currentLayer = outputLayer;
	while (currentLayer->GetInputLayer())
	{
		currentLayer->Train(this);
		currentLayer = currentLayer->GetInputLayer();
	}
}

Tensor Optimizer::GetBatch(const Tensor &original, unsigned int batchSize, unsigned int count)
{
	unsigned int matrixCount = original.GetMatrixCount();
	unsigned int firstMatrix = batchSize * count;
	unsigned int lastMatrix = batchSize * (count + 1);
	if (lastMatrix > matrixCount)
		lastMatrix = matrixCount;

	Tensor batch({original.GetShapeAt(0), original.GetShapeAt(1), lastMatrix - firstMatrix});

	return Tensor();
}
