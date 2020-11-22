#include "Optimizer.h"

Optimizer::Optimizer(std::shared_ptr<Layer> output)
{
	outputLayer = output;
}

Optimizer::~Optimizer()
{
}