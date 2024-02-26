#include <catch2/catch_all.hpp>

#include "NeuralNetwork/Layers/FeedForwardLayer.h"
#include "NeuralNetwork/Layers/InputLayer.h"
#include "NeuralNetwork/Optimizers/GradientDescent.h"
#include "NeuralNetwork/LossFunctions/MSE.hpp"

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

		GradientDescent gds(new MSE(), &outputLayer, 0.5f);

		WHEN("running the network")
		{
            float testInputValues[2] = {
                    0.05f, 0.05f
            };

            float expectedValues[2] = {
                    0.75136507f, 0.772928465f
            };

			Matrix testInput(1, 2, testInputValues);

			input.SetInput(testInput);
			Tensor result = outputLayer.ComputeAndGetOutput();

			Tensor expected({1, 2}, expectedValues);

			THEN("the difference is minimal")
			{
				Tensor diff = result - expected;
				REQUIRE(abs(diff.Sum()) < 0.01f);
			}
		}

		WHEN("training the network")
		{
            float testInputValues[2] = {
                    0.05f, 0.1f
            };

            float expectedValues[2] = {
                    0.01f, 0.99f
            };
			Tensor testInput({1, 2}, testInputValues);
			Tensor expected({1, 2}, expectedValues);

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