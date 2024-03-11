#include <catch2/catch_all.hpp>

#include "NeuralNetwork/Layers/LSTM.h"
#include "NeuralNetwork/Layers/InputLayer.h"
#include "NeuralNetwork/Optimizers/GradientDescent.h"
#include "NeuralNetwork/LossFunctions/MSE.hpp"

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
    GIVEN("a 3-output LSTM")
    {
        float WfValues[18] = {
                0.1f, 0.15f, 0.2f,
                0.2f, 0.25f, 0.3f,
                0.3f, 0.35f, 0.4f,
                0.4f, 0.45f, 0.5f,
                0.5f, 0.55f, 0.6f,
                0.6f, 0.65f, 0.7f
        };

        float BfValues[3] = {
                -0.1f, 0.1f, 0.2f
        };

        float WiValues[18] = {
                0.1f, 0.15f, 0.2f,
                0.2f, 0.25f, 0.3f,
                0.3f, 0.35f, 0.4f,
                0.4f, 0.45f, 0.5f,
                0.5f, 0.55f, 0.6f,
                0.6f, 0.65f, 0.7f
        };

        float BiValues[3] = {
                -0.1f, 0.1f, 0.2f
        };

        float WwValues[18] = {
                0.1f, 0.15f, 0.2f,
                0.2f, 0.25f, 0.3f,
                0.3f, 0.35f, 0.4f,
                0.4f, 0.45f, 0.5f,
                0.5f, 0.55f, 0.6f,
                0.6f, 0.65f, 0.7f
        };

        float BwValues[3] = {
                -0.1f, 0.1f, 0.2f
        };

        float WoValues[18] = {
                0.1f, 0.15f, 0.2f,
                0.2f, 0.25f, 0.3f,
                0.3f, 0.35f, 0.4f,
                0.4f, 0.45f, 0.5f,
                0.5f, 0.55f, 0.6f,
                0.6f, 0.65f, 0.7f
        };

        float BoValues[3] = {
                -0.1f, 0.1f, 0.2f
        };

        float inputValues[15] = {
                0, 0, 1,
                0, 1, 1,
                1, 1, 1,
                1, 0, 1,
                1, 1, 0
        };

        Matrix input(5, 3, inputValues);

        InputLayer inputLayer(3);
        LSTM lstm(&inputLayer, 3);

        inputLayer.SetInput(input);

        WHEN("calculating expected output")
        {
            Matrix state(1, 3);
            Matrix concated(1, 6);
            Matrix cell(1, 3);

            state.FillWith(0);
            cell.FillWith(0);

            Matrix Wf(6, 3, WfValues);
            Matrix Bf(1, 3, BfValues);

            Matrix Wi(6, 3, WiValues);
            Matrix Bi(1, 3, BiValues);
            Matrix Ww(6, 3, WwValues);
            Matrix Bw(1, 3, BwValues);

            Matrix Wo(6, 3, WoValues);
            Matrix Bo(1, 3, BoValues);

            lstm.GetWeight(LSTM::Gate::FORGET).Copy(Wf);
            lstm.GetBias(LSTM::Gate::FORGET).Copy(Bf);

            lstm.GetWeight(LSTM::Gate::INPUT).Copy(Wi);
            lstm.GetBias(LSTM::Gate::INPUT).Copy(Bi);

            lstm.GetWeight(LSTM::Gate::ACTIVATION).Copy(Ww);
            lstm.GetBias(LSTM::Gate::ACTIVATION).Copy(Bw);

            lstm.GetWeight(LSTM::Gate::OUTPUT).Copy(Wo);
            lstm.GetBias(LSTM::Gate::OUTPUT).Copy(Bo);

            Sigmoid sig = Sigmoid::GetInstance();
            TanhFunction tanh = TanhFunction::GetInstance();
            Softmax softmax = Softmax::GetInstance();

            Tensor calculatedOutput({5, 3}, nullptr);


            for (unsigned int i = 0; i < 5; i++)
            {
                TempMatrix inputRow = input.GetTempRowMatrix(i);
                concated = std::move(Matrix::Concat(state, inputRow, Matrix::ConcatType::BY_COLUMN));

                // forget
                Matrix forgetGate = concated * Wf;
                forgetGate += Bf;
                forgetGate = std::move(sig.CalculateMatrix(forgetGate));

                cell.ElementwiseMultiply(forgetGate);

                // input
                Matrix input1 = concated * Wi;
                Matrix input2 = concated * Ww;
                input1 += Bi;
                input2 += Bw;
                input1 = std::move(sig.CalculateMatrix(input1));
                input2 = std::move(tanh.CalculateMatrix(input2));

                input1.ElementwiseMultiply(input2);
                cell += input1;

                // output
                Matrix outputVector = tanh.CalculateMatrix(cell);
                Matrix stateUpdate = concated * Wo;
                stateUpdate += Bo;
                stateUpdate = std::move(sig.CalculateMatrix(stateUpdate));
                outputVector.ElementwiseMultiply(stateUpdate);
                state = std::move(outputVector);

                Matrix o = softmax.CalculateMatrix(state);
                o.CopyPartTo(calculatedOutput, 0, 3 * i, 3);
            }

            Tensor output = lstm.ComputeAndGetOutput();
            Tensor diff = calculatedOutput - output;
            diff.CopyFromGPU();

            REQUIRE(abs(diff.Sum()) < 0.001f);
        }
        WHEN("training the layer")
        {
            float expectedValues[15] = {
                    0, 1, 0,
                    1, 0, 0,
                    0, 1, 0,
                    0, 0, 1,
                    0, 0, 1
            };

            lstm.Reset();

            Tensor expectedOutput({5, 3}, expectedValues);
            Tensor firstOutput = lstm.ComputeAndGetOutput();

            MSE mse;
            float firstError = mse.Loss(firstOutput, expectedOutput);
            GradientDescent trainer(&mse, &lstm, 0.5f);

            Tensor inputTensor(input);
            trainer.Train(inputTensor, expectedOutput);

            lstm.Reset();
            Tensor secondOutput = lstm.ComputeAndGetOutput();
            float secondError = mse.Loss(secondOutput, expectedOutput);

            THEN("the second error is lower")
            {
                REQUIRE(secondError < firstError);
            }
        }
    }
}
