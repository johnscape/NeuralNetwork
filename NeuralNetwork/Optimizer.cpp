#include "Optimizer.h"

Optimizer::Optimizer(Layer& output)
{
	outputLayer.reset(&output);
}

Optimizer::~Optimizer()
{
}