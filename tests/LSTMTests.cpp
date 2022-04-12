#include <catch2/catch.hpp>

#include "NeuralNetwork/LSTM.h"
#include "NeuralNetwork/InputLayer.h"
#include "NeuralNetwork/GradientDescent.h"
#include "NeuralNetwork/LossFunctions.hpp"

/**
 * Layer test TODOs:
 * - Setting a value
 * - Getting the output with 2 pre-defined states
 * - Getting the GDS result
 */

SCENARIO("Setting an input for an LSTM layer", "[layer][init]")
{
	GIVEN("An LSTM layer with the size of 3")
	{
		InputLayer input(5);
		LSTM lstm(&input, 3);
		WHEN("Running a random input")
		{
			Tensor random({4, 5, 2});
			random.FillWithRandom();
			input.SetInput(random);
			Tensor output = lstm.ComputeAndGetOutput();

			THEN("The output will have consistent dimensions")
			{
				REQUIRE(output.GetShapeAt(0) == 4);
				REQUIRE(output.GetShapeAt(1) == 3);
				REQUIRE(output.GetShapeAt(2) == 2);
			}
		}
	}
}

SCENARIO("Training and running an LSTM layer", "[layer][training]")
{
	GIVEN("A 1-LSTM and 2 step input")
	{
		InputLayer input(2);
		LSTM lstm(&input, 1);

		lstm.GetWeight(0).SetValue(0, 0.1);
		lstm.GetWeight(0).SetValue(1, 0.7);
		lstm.GetWeight(0).SetValue(2, 0.45);

		lstm.GetWeight(1).SetValue(0, 0.8);
		lstm.GetWeight(1).SetValue(1, 0.95);
		lstm.GetWeight(1).SetValue(2, 0.8);

		lstm.GetWeight(2).SetValue(0, 0.15);
		lstm.GetWeight(2).SetValue(1, 0.45);
		lstm.GetWeight(2).SetValue(2, 0.25);

		lstm.GetWeight(3).SetValue(0, 0.25);
		lstm.GetWeight(3).SetValue(1, 0.6);
		lstm.GetWeight(3).SetValue(2, 0.4);

		lstm.GetBias(0).SetValue(0, 0.15);
		lstm.GetBias(1).SetValue(0, 0.65);
		lstm.GetBias(2).SetValue(0, 0.2);
		lstm.GetBias(3).SetValue(0, 0.1);

		Tensor inputValue({2, 2}, nullptr);
		inputValue.SetValue(0, 1);
		inputValue.SetValue(1, 2);
		inputValue.SetValue(2, 0.5);
		inputValue.SetValue(3, 3);

		Tensor expectedValues({2, 1}, nullptr);
		expectedValues.SetValue(0, 0.5);
		expectedValues.SetValue(1, 1.25);

		WHEN("Running the network")
		{
			input.SetInput(inputValue);
			Tensor output = lstm.ComputeAndGetOutput();

			THEN("The difference between the expected one is negligible")
			{
				REQUIRE(abs(output.GetValue(0) - 0.53631) < 0.001);
				REQUIRE(abs(output.GetValue(1) - 0.77197) < 0.001);
			}
		}
		WHEN("training the network")
		{
			GradientDescent trainer(LossFunctions::MSE, LossFunctions::MSE_Derivate, &lstm, 0.1);
			trainer.Train(inputValue, expectedValues);
		}
	}
}
