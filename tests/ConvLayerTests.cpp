#include <catch2/catch.hpp>
#include "NeuralNetwork/Layers/ConvLayer.h"
#include "NeuralNetwork/Layers/InputLayer.h"

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