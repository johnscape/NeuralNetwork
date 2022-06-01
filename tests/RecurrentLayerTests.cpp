#include <catch2/catch.hpp>

#include "NeuralNetwork/Layers/RecurrentLayer.h"
#include "NeuralNetwork/Layers/InputLayer.h"
#include "NeuralNetwork/ActivationFunctions.hpp"
#include "NeuralNetwork/Optimizers/GradientDescent.h"
#include "NeuralNetwork/LossFunctions/MSE.hpp"

#include <cmath>

SCENARIO("Setting an input for a recurrent layer", "[layer][init]")
{
	GIVEN("a recurrent layer with the size of 8")
	{
		InputLayer inputLayer(5);
		RecurrentLayer recurrentLayer(&inputLayer, 8);
		WHEN("giving it an 15x5x23 sized input")
		{
			Tensor input({15, 5, 23});
			input.FillWithRandom();
			inputLayer.SetInput(input);
			THEN("the output is sized 15x8x23")
			{
				Tensor result = recurrentLayer.ComputeAndGetOutput();
				REQUIRE(result.GetShapeAt(0) == 15);
				REQUIRE(result.GetShapeAt(1) == 8);
				REQUIRE(result.GetShapeAt(2) == 23);
			}
		}
	}
}

SCENARIO("Running a recurrent layer", "[layer][computation]")
{
	GIVEN("a recurrent network with TANH function and a size of 3")
	{
		InputLayer inputLayer(3);
		RecurrentLayer recurrentLayer(&inputLayer, 3);
		recurrentLayer.SetActivationFunction(ACTIVATION_TANH);

		Matrix recWeight(3, 3);
		Matrix normWeight(3, 3);

		recWeight = Matrix::Eye(3);
		normWeight.FillWith(0.1f);

		recurrentLayer.GetBias().FillWith(0);

		recurrentLayer.GetWeights().ReloadFromOther(normWeight);
		recurrentLayer.GetRecurrentWeights().ReloadFromOther(recWeight);

		WHEN("running a tensor of 3x3 trough the system")
		{
			Tensor input({3, 3}, nullptr);
			input.FillWith(1);
			inputLayer.SetInput(input);
			Tensor output = recurrentLayer.ComputeAndGetOutput();
			THEN("the output should be good in shape")
			{
				REQUIRE(output.GetShapeAt(0) == 3);
				REQUIRE(output.GetShapeAt(1) == 3);
			}
			THEN("the output should be good")
			{
				Tensor expected({3, 3}, nullptr);
				expected.SetValue(0, 0.291312f);
				expected.SetValue(1, 0.291312f);
				expected.SetValue(2, 0.291312f);

				expected.SetValue(3, 0.537049f);
				expected.SetValue(4, 0.537049f);
				expected.SetValue(5, 0.537049f);

				expected.SetValue(6, 0.716297f);
				expected.SetValue(7, 0.716297f);
				expected.SetValue(8, 0.716297f);

				Tensor res = expected - output;
				REQUIRE(abs(res.Sum()) < 0.01f);

			}
		}
	}
}

SCENARIO("Training a recurrent layer", "[layer][training]")
{
	GIVEN("A predictable RNN and a 4x2 input")
	{
		InputLayer inputLayer(2);
		RecurrentLayer rnn(&inputLayer, 2);

		for (unsigned int i = 0; i < 4; i++)
		{
			rnn.GetWeights().SetValue(i, 1);
			rnn.GetRecurrentWeights().SetValue(i, 2);
			rnn.GetBias().SetValue(i % 2, 1);
		}

		rnn.SetActivationFunction(ACTIVATION_LINEAR);

		Tensor input({4, 2}, nullptr);

		input.SetValue(0, 0);
		input.SetValue(1, 0);
		input.SetValue(2, 1);
		input.SetValue(3, 0);
		input.SetValue(4, 0);
		input.SetValue(5, 1);
		input.SetValue(6, 1);
		input.SetValue(7, 1);

		Tensor expected({4, 2}, nullptr);

		expected.SetValue(0, 0);
		expected.SetValue(1, 0);
		expected.SetValue(2, 1);
		expected.SetValue(3, 0);
		expected.SetValue(4, 0);
		expected.SetValue(5, 0);
		expected.SetValue(6, 0);
		expected.SetValue(7, 1);

		GradientDescent trainer(new MSE(), &rnn, 0.5f);

		WHEN("running the network")
		{
			inputLayer.SetInput(input);
			Tensor result = rnn.ComputeAndGetOutput();

			Tensor output({4, 2}, nullptr);


			output.SetValue(0, 1);
			output.SetValue(1, 1);
			output.SetValue(2, 6);
			output.SetValue(3, 6);
			output.SetValue(4, 26);
			output.SetValue(5, 26);
			output.SetValue(6, 107);
			output.SetValue(7, 107);

			Tensor diff = output - result;
			REQUIRE(abs(diff.Sum()) < 0.01f);
		}

		WHEN("training the network")
		{
			trainer.Train(input, expected);

			Matrix wantedWeight(2, 2);
			Matrix wantedRecWeight(2, 2);

			wantedWeight.SetValue(0, 1 - 112 * 0.25 * 0.5);
			wantedWeight.SetValue(2, 1 - 133 * 0.25 * 0.5);
			wantedWeight.SetValue(1, 1 - 112 * 0.25 * 0.5);
			wantedWeight.SetValue(3, 1 - 132 * 0.25 * 0.5);

			wantedRecWeight.SetValue(0, 2 - 915.875f);
			wantedRecWeight.SetValue(2, 2 - 915.875f);
			wantedRecWeight.SetValue(1, 2 - 907.75f);
			wantedRecWeight.SetValue(3, 2 - 907.75f);


			Matrix weightDiff = wantedWeight - rnn.GetWeights();
			Matrix recDiff = wantedRecWeight - rnn.GetRecurrentWeights();

			REQUIRE(abs(weightDiff.Sum()) < 0.001f);
			REQUIRE(abs(recDiff.Sum()) < 0.001f);
		}
	}
}