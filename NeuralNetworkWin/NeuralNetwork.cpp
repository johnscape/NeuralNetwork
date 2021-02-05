#include <iostream>
#include "Matrix.h"
#include "MatrixMath.h"
#include "Model.h"
#include "FeedForwardLayer.h"
#include "InputLayer.h"
#include "GradientDescent.h"
#include "LossFunctions.hpp"

#include "MatrixGPUMath.cuh"

#include "Model.h"

#include <chrono>

int main()
{

	srand(time(0));

	Model* m1 = new Model();
	Model* m2;
	Model m3;

	InputLayer* inp = new InputLayer(5);
	FeedForwardLayer* hidden = new FeedForwardLayer(inp, 15);
	FeedForwardLayer* out = new FeedForwardLayer(hidden, 8);

	m1->AddLayer(inp);
	m1->AddLayer(hidden);
	m1->AddLayer(out);

	Matrix test(1, 5);
	for (unsigned int i = 0; i < 5; i++)
	{
		test.SetValue(i, (float)i / 5);
	}

	test.CopyToGPU();

	MatrixMath::PrintMatrix(&test);

	m2 = new Model(*m1);
	m3 = *m1;


	MatrixMath::PrintMatrix(m1->Compute(&test));
	MatrixMath::PrintMatrix(m2->Compute(&test));
	MatrixMath::PrintMatrix(m3.Compute(&test));

	return 0;
}
