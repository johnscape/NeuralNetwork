#include <iostream>
#include <stdlib.h>
#include <time.h> 
#include "Model.h"
#include "InputLayer.h"
#include "FeedForwardLayer.h"
#include "GeneticAlgorithm.h"
#include "Matrix.h"
#include "MatrixMath.h"
#include "LossFunctions.hpp"
#include "ActivationFunctions.hpp"

float Tester(Model* model)
{
	float error = 0;
	//srand(time(0));

	/*Matrix input(1, 16);
	for (unsigned char i = 0; i < 16; i++)
		if (rand() % 2 == 1)
			input.SetValue(i, 1);
	Matrix expected(1, 8);
	*/

	Matrix input(1, 2);
	Matrix expected(1, 1);

	Matrix* output;


	for (unsigned int p = 0; p < 50; p++)
	{
		input.SetValue(0, rand() % 2 ? 1 : 0);
		input.SetValue(1, rand() % 2 ? 1 : 0);

		/*if (input.GetValue(0) == 1 && input.GetValue(1) == 0)
			expected.SetValue(0, 1);
		else if (input.GetValue(0) == 0 && input.GetValue(1) == 1)
			expected.SetValue(0, 1);
		else
			expected.SetValue(0, 0);*/
		if (input.GetValue(0) + input.GetValue(1) == 1)
			expected.SetValue(0, 1);
		else
			expected.SetValue(0, 0);

		input.CopyToGPU();
		output = model->Compute(&input);

		float err = LossFunctions::MSE(output, &expected);


		error += err;
	}
	
	error /= 50;
	std::cout << "Model output was " << output->GetValue(0) << " expected was " << expected.GetValue(0) << " average error is " << error << std::endl;

	//std::cout << "Fitness is " << error << std::endl;

	return error;
}

int main()
{
	srand(time(0));
	Model m;
	m.AddLayer(new InputLayer(2));
	m.AddLayer(new FeedForwardLayer(m.GetLastLayer(), 2));
	m.AddLayer(new FeedForwardLayer(m.GetLastLayer(), 1));

	dynamic_cast<FeedForwardLayer*>(m.GetLastLayer())->SetActivationFunction(&(Sigmoid::GetInstance()));

	//GeneticAlgorithm trainer(&m, 50, 500, &Tester);
	GeneticAlgorithm trainer(&m, 500, 10, &Tester);
	trainer.Train(nullptr, nullptr);

	return 0;
}
