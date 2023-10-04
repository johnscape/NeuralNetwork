#include <catch2/catch_all.hpp>

#include "NeuralNetwork/Layers/InputLayer.h"

/**
 * Layer test TODOs:
 * - Setting a value
 * - Giving an input
 * - Getting the output with 2 pre-defined states
 * - Getting the GDS result
 */

SCENARIO("Testing input layer's input", "[layer]")
{
	GIVEN("an input layer with the size of 5")
	{
		InputLayer input(5);
		WHEN("setting the input of a 1x5 matrix")
		{
			Matrix m(1, 5);
			input.SetInput(m);
			THEN("the output is the same")
			{
				REQUIRE(input.GetOutput() == m);
			}
		}
		WHEN("setting the input of a 3x5x9 tensor")
		{
			Tensor t({3, 5, 9});
			input.SetInput(t);
			THEN("the output is the same")
			{
				REQUIRE(t == input.GetOutput());
			}
		}
	}
}