#include <catch2/catch.hpp>

#include "NeuralNetwork/FeedForwardLayer.h"
#include "NeuralNetwork/InputLayer.h"
#include "NeuralNetwork/ActivationFunctions.hpp"
#include "NeuralNetwork/GradientDescent.h"
#include "NeuralNetwork/LossFunctions.hpp"

#include <cmath>
#include <fstream>

/**
 * Layer test TODOs:
 * - Setting a value
 * - Getting the output with 2 pre-defined states
 * - Getting the GDS result
 */

SCENARIO("Setting an input to the Feed Foward Layer", "[layer][init]")
{
	GIVEN("a feed forward and an input layer")
	{
		InputLayer input(5);
		FeedForwardLayer layer(&input, 5);
		layer.SetActivationFunction(&IdentityFunction::GetInstance());
		layer.GetBias().FillWith(0);

		WHEN("setting the layer's weight to identity")
		{
			layer.GetWeights().ReloadFromOther(Matrix::Eye(5));
			THEN("the output will be the same as the input")
			{
				Tensor i({5, 5, 5});
				i.FillWithRandom();
				input.SetInput(i);
				Tensor o = layer.ComputeAndGetOutput();
				REQUIRE(abs((o - i).Sum())< 0.1f);
			}
		}
	}
}

SCENARIO("Getting the output from the Feed Forward Layer", "[layer][computation]")
{
	GIVEN("a feed forward network with sigmoid activation")
	{
		InputLayer inputLayer(5);
		FeedForwardLayer feedForwardLayer(&inputLayer, 8);
		WHEN("setting the input as a 1x5x6 tensor filled with 0.5")
		{
			Tensor tensor({1, 5, 6});
			tensor.FillWith(0.5f);
			inputLayer.SetInput(tensor);
			THEN("the output is the sig(input*weight+bias)")
			{
				Tensor inner = tensor * feedForwardLayer.GetWeights();
				inner += feedForwardLayer.GetBias();
				Tensor result = TanhFunction::GetInstance().CalculateTensor(inner);
				Tensor output = feedForwardLayer.ComputeAndGetOutput();
				REQUIRE(result == output);
			}
		}
	}
}

SCENARIO("Training the feed forward layer", "[layer][training]")
{
	GIVEN("a 2-input layer, a 2x2 feedforward layer and a gradient descent trainer")
	{
		InputLayer input(2);
		FeedForwardLayer hiddenLayer(&input, 2);

		hiddenLayer.GetWeights().SetValue(0, 0.15f);
		hiddenLayer.GetWeights().SetValue(1, 0.25f);
		hiddenLayer.GetWeights().SetValue(2, 0.2f);
		hiddenLayer.GetWeights().SetValue(3, 0.3f);

		hiddenLayer.GetBias().SetValue(0, 0.35f);
		hiddenLayer.GetBias().SetValue(1, 0.35f);

		hiddenLayer.SetActivationFunction(ACTIVATION_SIGMOID);

		FeedForwardLayer outputLayer(&hiddenLayer, 2);

		outputLayer.GetWeights().SetValue(0, 0.4f);
		outputLayer.GetWeights().SetValue(1, 0.5f);
		outputLayer.GetWeights().SetValue(2, 0.45f);
		outputLayer.GetWeights().SetValue(3, 0.55f);

		outputLayer.GetBias().SetValue(0, 0.6f);
		outputLayer.GetBias().SetValue(1, 0.6f);

		outputLayer.SetActivationFunction(ACTIVATION_SIGMOID);

		GradientDescent gds(LossFunctions::MSE, LossFunctions::MSE_Derivate, &outputLayer, 0.5f);

		WHEN("running the network")
		{
			Matrix testInput(1, 2);
			testInput.SetValue(0, 0.05f);
			testInput.SetValue(1, 0.1f);

			input.SetInput(testInput);
			Matrix result = (Matrix)outputLayer.ComputeAndGetOutput();

			Matrix expected(1, 2);
			expected.SetValue(0, 0.75136507f);
			expected.SetValue(1, 0.772928465f);

			THEN("the difference is minimal")
			{
				Matrix diff = result - expected;
				REQUIRE(abs(diff.Sum()) < 0.01f);
			}
		}

		WHEN("training the network")
		{
			Tensor testInput({1, 2}, nullptr);
			testInput.SetValue(0, 0.05f);
			testInput.SetValue(1, 0.1f);

			Tensor expected({1, 2}, nullptr);
			expected.SetValue(0, 0.01f);
			expected.SetValue(1, 0.99f);

			input.SetInput(testInput);
			Tensor result = outputLayer.ComputeAndGetOutput();

			float error = LossFunctions::MSE(result, expected);
			REQUIRE(abs(error - 0.298371109) < 0.01);
			gds.Train(testInput, expected);

			Matrix wantedOutputWeights(2, 2);
			wantedOutputWeights.SetValue(0, 0.35891648f);
			wantedOutputWeights.SetValue(1, 0.408666186f);
			wantedOutputWeights.SetValue(2, 0.511301270f);
			wantedOutputWeights.SetValue(3, 0.561370121f);

			Matrix wantedHiddenWeights(2, 2);
			wantedHiddenWeights.SetValue(0, 0.149780716f);
			wantedHiddenWeights.SetValue(1, 0.19956143f);
			wantedHiddenWeights.SetValue(2, 0.24975114f);
			wantedHiddenWeights.SetValue(3, 0.29950229f);

			Matrix outputDiff = wantedOutputWeights - outputLayer.GetWeights();
			Matrix hiddenDiff = wantedHiddenWeights - hiddenLayer.GetWeights();

			REQUIRE(abs(outputDiff.Sum()) < 0.01f);
			REQUIRE(abs(hiddenDiff.Sum()) < 0.01f);

		}
	}
}