#include "Optimizer.h"
#include "MatrixMath.h"

Optimizer::Optimizer(Layer* output) : currentBatch(0), lastError(0)
{
	outputLayer = output;
	inputLayer = nullptr;
}

Optimizer::~Optimizer()
{
}

void Optimizer::TrainFor(Matrix* input, Matrix* expected, unsigned int times, unsigned int batch)
{
	FindInputLayer();
	for (unsigned int time = 0; time < times; time++)
	{
		currentBatch = 0;
		for (unsigned int pos = 0; pos < input->GetRowCount(); pos++)
		{
			Matrix* inp = MatrixMath::GetRowMatrix(input, pos);
			Matrix* exp = MatrixMath::GetRowMatrix(expected, pos);
			TrainStep(inp, exp);
			currentBatch++;
			if (currentBatch >= batch)
			{
				TrainLayers();
				currentBatch = 0;
			}

			delete inp;
			delete exp;
		}
		if (currentBatch > 0)
			TrainLayers();
	}
}

void Optimizer::TrainUntil(Matrix* input, Matrix* expected, float error, unsigned int batch)
{
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
