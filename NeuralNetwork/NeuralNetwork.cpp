#include <iostream>
#include "Matrix.h"
#include "InputLayer.h"
#include "FeedForwardLayer.h"
#include "LSTM.h"
#include "ActivationFunctions.hpp"
#include "Constants.h"

#include "GradientDescent.h"
#include "LossFunctions.hpp"

#include <memory>
int main()
{
	InputLayer inp(2);
	LSTM lstm(&inp, 1);

	lstm.GetWeight(2)->SetValue(0, 0.45f);
	lstm.GetWeight(2)->SetValue(1, 0.25f);
	lstm.GetWeight(1)->SetValue(0, 0.95f);
	lstm.GetWeight(1)->SetValue(1, 0.8f);
	lstm.GetWeight(0)->SetValue(0, 0.7f);
	lstm.GetWeight(0)->SetValue(1, 0.45f);
	lstm.GetWeight(3)->SetValue(0, 0.6f);
	lstm.GetWeight(3)->SetValue(1, 0.4f);

	lstm.GetRecursiveWeight(2)->SetValue(0, 0.15f);
	lstm.GetRecursiveWeight(1)->SetValue(0, 0.8f);
	lstm.GetRecursiveWeight(0)->SetValue(0, 0.1f);
	lstm.GetRecursiveWeight(3)->SetValue(0, 0.25f);

	lstm.GetBias(2)->SetValue(0, 0.2f);
	lstm.GetBias(1)->SetValue(0, 0.65f);
	lstm.GetBias(0)->SetValue(0, 0.15f);
	lstm.GetBias(3)->SetValue(0, 0.1f);

	Matrix input(2, 2);
	input.SetValue(0, 1);
	input.SetValue(1, 2);
	input.SetValue(2, 0.5f);
	input.SetValue(3, 3);

	Matrix expected(2, 1);
	expected.SetValue(0, 0.5f);
	expected.SetValue(1, 1.25f);

	GradientDescent trainer(LossFunctions::MSE, LossFunctions::MSE_Derivate, &lstm, 0.5);

	trainer.TrainFor(&input, &expected, 10, 1);

	return 0;
}
