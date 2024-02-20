#include <catch2/catch_all.hpp>

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
                expected.CopyToGPU();
				Tensor res = expected - output;
                res.CopyFromGPU();
				REQUIRE(abs(res.Sum()) < 0.01f);

			}
		}
	}
}

SCENARIO("Training a recurrent layer", "[layer][training]")
{
	GIVEN("A predictable RNN and a 4x2 input")
	{
        // test data
        float inputValues[10] = {
                0, 0,
                0, 1,
                1, 0,
                1, 1,
                0, 1
        };

        float WVals[4] = {
                0.1f, 0.2f,
                0.3f, 0.4f
        };

        float BVals[2] = {
                0.1f, 0.2f
        };

        float RVals[4] = {
                0.4f, 0.5f,
                0.5f, 0.4f
        };

        float expectedValues[10] = {
                0, 0,
                0, 0,
                0, 0,
                1, 0,
                0, 1
        };

        Tensor input({5, 2}, inputValues);
        Matrix W(2, 2, WVals);
        Matrix R(2, 2, RVals);
        Matrix B(1, 2, BVals);
        Tensor expected({5, 2}, expectedValues);

		InputLayer inputLayer(2);
		RecurrentLayer rnn(&inputLayer, 2);

        rnn.GetWeights().Copy(W);
        rnn.GetRecurrentWeights().Copy(R);
        rnn.GetBias().Copy(B);

		rnn.SetActivationFunction(ACTIVATION_LINEAR);

		GradientDescent trainer(new MSE(), &rnn, 0.5f);

		WHEN("running the network")
		{
			inputLayer.SetInput(input);
			Tensor result = rnn.ComputeAndGetOutput();
            result.CopyFromGPU();

            float calculatedOutputVale[10] = {
                    0.1f, 0.2f,
                    0.54f, 0.73f,
                    0.781f, 0.962f,
                    1.2934f, 1.5753f,
                    1.70501f, 1.87682f
            };
			Tensor output({5, 2}, calculatedOutputVale);

			Tensor diff = output - result;
            diff.CopyFromGPU();
			REQUIRE(abs(diff.Sum()) < 0.01f);
		}

		WHEN("training the network")
		{
			trainer.Train(input, expected);

            float targetWeightValues[4] = {
                    0.0565f, -0.0053f,
                    0.0871f, 0.1635f
            };

            float targetRecurrentWeightValues[4] = {
                    0.3403f, 0.4099f,
                    0.4408f, 0.31085f
            };

            float targetBiasValues[2] = {
                    -0.1497f, -0.0968f
            };

			Matrix targetWeight(2, 2, targetWeightValues);
			Matrix targetRecurrentWeight(2, 2, targetRecurrentWeightValues);
            Matrix targetBias(1, 2, targetBiasValues);


			Matrix weightDiff = targetWeight - rnn.GetWeights();
			Matrix recDiff = targetRecurrentWeight - rnn.GetRecurrentWeights();
            Matrix biasDiff = targetBias - rnn.GetBias();

			REQUIRE(abs(weightDiff.Sum()) < 0.001f);
			REQUIRE(abs(recDiff.Sum()) < 0.001f);
            REQUIRE(abs(biasDiff.Sum()) < 0.001f);
		}
	}
}