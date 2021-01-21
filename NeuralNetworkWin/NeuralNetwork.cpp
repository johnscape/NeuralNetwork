#include <iostream>
#include "Matrix.h"
#include "MatrixMath.h"
#include "Model.h"
#include "LSTM.h"
#include "InputLayer.h"
#include "GradientDescent.h"
#include "LossFunctions.hpp"

#include "MatrixGPUMath.cuh"

#include <chrono>

int main()
{
	/*Matrix a(5, 5);
	Matrix b(5, 5);
	MatrixMath::FillWith(&a, 5.0f);
	MatrixMath::FillWith(&b, 3.0f);

	MatrixMath::Copy(&a, &b);
	b.CopyFromGPU();
	MatrixMath::PrintMatrix(&b);*/


	Matrix input(1, 5);
	for (size_t i = 0; i < 5; i++)
	{
		input.SetValue(i, (float)(i + 1) / 10);
	}

	Matrix output(1, 15);
	for (size_t i = 0; i < 15; i++)
	{
		output.SetValue(i, (float)(i + 1) / 20);
	}

	input.CopyToGPU();
	output.CopyToGPU();

	Model m;
	m.LoadModel("model_1.json");
	GradientDescent descent(LossFunctions::MSE, LossFunctions::MSE_Derivate, m.GetOutput(), 0.5);
	descent.TrainStep(&input, &output);
	Matrix* out = m.Compute(&input);
	out->CopyFromGPU();
	MatrixMath::PrintMatrix(out);
	Matrix out_old;
	out_old.LoadFromJSON("output.json", true);
	std::cout << MatrixMath::IsEqual(out, &out_old) << std::endl;
	MatrixMath::PrintMatrix(&out_old);

	/*input.SaveToJSON("input.json");
	Model m;
	m.AddLayer(new InputLayer(5));
	m.AddLayer(new LSTM(m.GetLastLayer(), 15));

	GradientDescent descent(LossFunctions::MSE, LossFunctions::MSE_Derivate, m.GetOutput(), 0.5);
	m.SaveModel("model_1.json");
	descent.TrainStep(&input, &output);
	m.SaveModel("model_2.json");
	Matrix* out = m.Compute(&input);
	out->SaveToJSON("output.json");*/



	return 0;
}
