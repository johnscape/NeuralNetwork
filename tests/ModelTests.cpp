#include <catch2/catch_all.hpp>
#include "NeuralNetwork/Model.h"
#include "NeuralNetwork/Layers/InputLayer.h"
#include "NeuralNetwork/Layers/FeedForwardLayer.h"
#include "NeuralNetwork/Optimizers/GradientDescent.h"
#include "NeuralNetwork/LossFunctions/MSE.hpp"

/**
 * Needed tests:
 * - Run model
 * - Train model
 * - Save model
 * - Load model
*/

SCENARIO("Creating a model", "[model][init]")
{
	GIVEN("an empty model and a test input tensor")
	{
		Model model;
		Tensor tensor({3, 5}, nullptr);
		tensor.FillWith(1);

		WHEN("checking properties")
		{
			THEN("the model has 0 Layers")
			{
				REQUIRE(model.GetLayerCount() == 0);
			}
			THEN("the model input and output is nullpr")
			{
				REQUIRE(model.GetInput() == nullptr);
				REQUIRE(model.GetOutput() == nullptr);
			}
			THEN("the model cannot be run")
			{
				REQUIRE_THROWS(model.Compute(tensor));
			}
			THEN("the model cannot be trained")
			{
				REQUIRE_THROWS(model.Train(nullptr));
			}
		}
		WHEN("adding a few Layers to the model")
		{
			InputLayer* inputLayer = new InputLayer(5);
			FeedForwardLayer* outputLayer = new FeedForwardLayer(inputLayer, 8);

			model.AddLayer(inputLayer);
			model.AddLayer(outputLayer);

			THEN("the model has 2 Layers")
			{
				REQUIRE(model.GetLayerCount() == 2);
			}
			THEN("the model's input is the input layer")
			{
				REQUIRE(model.GetInput() == inputLayer);
			}
			THEN("the model's output is the output layer")
			{
				REQUIRE(model.GetOutput() == outputLayer);
			}
			THEN("the model's first layer is the input, the second is the output")
			{
				REQUIRE(model.GetLayerAt(0) == inputLayer);
				REQUIRE(model.GetLayerAt(1) == outputLayer);
			}
		}
	}
}

SCENARIO("Running the model", "[model][computation]")
{
	GIVEN("a two layer model")
	{
		InputLayer* inputLayer = new InputLayer(5);
		FeedForwardLayer* outputLayer = new FeedForwardLayer(inputLayer, 3);

		float weightValues[15] = {
				0.5, 0, 0.5,
				0.5, 0.5, 0,
				0.5, 0, 0.5,
				0.5, 0.5, 0,
				0.5, 0, 0.5
		};

		Matrix tmp(5, 3, weightValues);

		outputLayer->GetWeights().ReloadFromOther(tmp);
		outputLayer->GetBias().FillWith(0);

		Model model;
		model.AddLayer(inputLayer);
		model.AddLayer(outputLayer);

		WHEN("getting an output")
		{
			float values[5] = {0, 1, 0, 1, 0};
			Tensor input({1, 5}, values);

			Tensor output = model.Compute(input);
			THEN("the result should be the following")
			{
				float s1 = 0.7310586;
				float outputValue[3] = {s1, s1, 0.5};
				Tensor expected({1, 3}, outputValue);
				Tensor diff = output - expected;

				REQUIRE(abs(diff.Sum()) < 0.0001);
			}
		}
	}
}

SCENARIO("Training the model", "[model][training]")
{
	GIVEN("a two layer model")
	{
		InputLayer* inputLayer = new InputLayer(5);
		FeedForwardLayer* outputLayer = new FeedForwardLayer(inputLayer, 3);

		float weightValues[15] = {
				0.5, 0, 0.5,
				0.5, 0.5, 0,
				0.5, 0, 0.5,
				0.5, 0.5, 0,
				0.5, 0, 0.5
		};

		Matrix tmp(5, 3, weightValues);

		outputLayer->GetWeights().ReloadFromOther(tmp);
		outputLayer->GetBias().FillWith(0);

		Model model;
		model.AddLayer(inputLayer);
		model.AddLayer(outputLayer);

		WHEN("training the model")
		{
			float values[5] = {0, 1, 0, 1, 0};
			Tensor input({1, 5}, values);

			float expected[3] = {1, 0, 1};
			Tensor expectedOutput({1, 3}, expected);

			Tensor output1 = model.Compute(input);
			MSE errorFunc;
			float error1 = errorFunc.Loss(output1, expectedOutput);

			GradientDescent gradientDescent(&errorFunc, &model, 0.5);
			gradientDescent.Train(input, expectedOutput);
			Tensor output2 = model.Compute(input);
			float error2 = errorFunc.Loss(output2, expectedOutput);
			REQUIRE(error2 < error1);
		}
	}
}

SCENARIO("Saving and loading the model", "[model][io]")
{
	GIVEN("a 3 layer model")
	{
		InputLayer inputLayer(5);
		FeedForwardLayer hiddenLayer(&inputLayer, 8);
		FeedForwardLayer outputLayer(&hiddenLayer, 3);

		Model model;
		model.AddLayer(&inputLayer);
		model.AddLayer(&hiddenLayer);
		model.AddLayer(&outputLayer);

		WHEN("saving the model and loading it into another")
		{
			std::string jsonText = model.SaveToString();
			Model model2;
			model2.LoadFromString(jsonText);

			THEN("the two model has to be the same")
			{
				REQUIRE(model2.GetLayerCount() == model.GetLayerCount());
				//TODO: Add more requirements
			}
		}
	}
}