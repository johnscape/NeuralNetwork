#include <catch2/catch.hpp>
#include "NeuralNetwork/Layers/ConvLayer.h"
#include "NeuralNetwork/Layers/InputLayer.h"
#include "NeuralNetwork/Optimizers/GradientDescent.h"
#include "NeuralNetwork/LossFunctions/MSE.hpp"

/**
 * Layer test TODOs:
 * - Setting a value
 * - Getting the output with 2 pre-defined states
 * - Getting the GDS result
 */

SCENARIO("Setting an input to the Conv Layer", "[layer][init]")
{
	GIVEN("a conv layer 1x1 and an input layer")
	{
		InputLayer input({3, 3});
		ConvLayer conv(&input, 1, 1);

		WHEN("checking the initial values")
		{
			THEN("the pad size is 0")
			{
				REQUIRE(conv.GetPadSize() == 0);
			}
			THEN("the kernel is 1x1x1x1")
			{
				REQUIRE(conv.GetKernel().GetShape().size() == 4);
				REQUIRE(conv.GetKernel().GetShapeAt(0) == 1);
				REQUIRE(conv.GetKernel().GetShapeAt(1) == 1);
				REQUIRE(conv.GetKernel().GetShapeAt(2) == 1);
				REQUIRE(conv.GetKernel().GetShapeAt(3) == 1);
			}
		}
	}
}

SCENARIO("Getting the output from the Conv Layer", "[layer][computation]")
{
	GIVEN("a basic conv layer and an input layer")
	{
		InputLayer input({3, 3});
		ConvLayer conv(&input, 1, 1);

		conv.GetKernel().FillWith(1);
		WHEN("running this network")
		{
			Tensor inp({3, 3}, nullptr);
			inp.FillWithRandom(0, 1);
			input.SetInput(inp);
			Tensor output = conv.ComputeAndGetOutput();
			output.Squeeze();
			THEN("the output has to be the same as the input")
			{
				REQUIRE(output.GetShape() == inp.GetShape());
				REQUIRE(output == inp);
			}
		}
	}
}

SCENARIO("Training the conv layer", "[layer][training]")
{
	GIVEN("a basic conv layer")
	{
		InputLayer inputLayer({3, 3, 3});
		ConvLayer convLayer(&inputLayer, 2, 1);

		convLayer.GetKernel().FillWith(1);

		float values[3*3*3] = {
				0, 1, 1,
				0, 0, 0,
				0, 0, 0,

				-1, -1, -1,
				0, 0, 0,
				1, 1, 1,

				-1, 0, 1,
				-1, 0, 1,
				-1, 0, 1
		};

		Tensor inputTensor({3, 3, 3}, values);
		inputLayer.SetInput(inputTensor);

		WHEN("running the network")
		{
			Tensor output = convLayer.ComputeAndGetOutput();
			output.Squeeze();
			Tensor expected({2, 2}, nullptr);
			expected.SetValue(0, 0);
			expected.SetValue(1, 2);
			expected.SetValue(2, 0);
			expected.SetValue(3, 4);

			THEN("the result is expected")
			{
				Tensor diff = expected - output;
				REQUIRE(abs(diff.Sum()) < 0.001f);
			}
		}

		WHEN("training the network")
		{
			GradientDescent gds(new MSE(), &convLayer, 0.5);
			float expectedValues[4] = {0, 1, 0, 0};
			Tensor expected({2, 2, 1}, expectedValues);

			float expectedKernelValues[12] = {
					0.5, 0.5,
					1, 1,

					1.5, 1.5,
					-1, -1,

					1, -1.5,
					1, -1.5
			};
			Tensor expectedKernel({2, 2, 3, 1}, expectedKernelValues);

			gds.Train(inputTensor, expected, 3);

			THEN("the new kernel will be this")
			{
				Tensor diff = convLayer.GetKernel() - expectedKernel;
				REQUIRE(abs(diff.Sum()) < 0.001);
			}
		}
	}
}