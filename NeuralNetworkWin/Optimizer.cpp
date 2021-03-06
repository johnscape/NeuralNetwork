#include "Optimizer.h"
#include "MatrixMath.h"

Optimizer::Optimizer(Layer* output) : currentBatch(0), lastError(0)
{
	outputLayer = output;
	//inputLayer = nullptr;
	FindInputLayer();
}

Optimizer::~Optimizer()
{
}

void Optimizer::TrainFor(Matrix* input, Matrix* expected, unsigned int times, unsigned int batch)
{
	FindInputLayer();
	SetTrainingMode(true);
	for (unsigned int time = 0; time < times; time++)
	{
		currentBatch = 0;
		for (unsigned int pos = 0; pos < input->GetRowCount(); pos++)
		{
			Matrix* inp = MatrixMath::GetRowMatrix(input, pos);
			Matrix* exp = MatrixMath::GetRowMatrix(expected, pos);
			currentBatch++;
			TrainStep(inp, exp);
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
	SetTrainingMode(false);
}

void Optimizer::TrainUntil(Matrix* input, Matrix* expected, float error, unsigned int batch)
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
