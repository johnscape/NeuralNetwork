#include <iostream>
#include "Model.h"
#include "InputLayer.h"
#include "FeedForwardLayer.h"

int main()
{
	/*FeedForwardLayer layer(nullptr, 5);
	MatrixMath::FillWithRandom(layer.GetWeights());
	MatrixMath::FillWithRandom(layer.GetBias());
	layer.SaveToJSON("layer.json");

	FeedForwardLayer layer2(nullptr, 5);
	layer2.LoadFromJSON("layer.json", true);

	MatrixMath::PrintMatrix(layer.GetWeights());*/
	Model model;
	model.AddLayer(new InputLayer(3));
	model.AddLayer(new FeedForwardLayer(model.GetLastLayer(), 3));
	model.AddLayer(new FeedForwardLayer(model.GetLastLayer(), 5));

	Matrix test(1, 3);
	MatrixMath::FillWithRandom(&test);
	std::cout << "The input: " << std::endl;
	MatrixMath::PrintMatrix(&test);

	Matrix out1 = model.Compute(&test);
	std::cout << "Return of the first model: " << std::endl;
	MatrixMath::PrintMatrix(&out1);

	model.SaveModel("model.json");

	Model model2;
	model2.LoadModel("model.json");

	std::cout << "Return of the second model: " << std::endl;
	Matrix out2 = model2.Compute(&test);
	MatrixMath::PrintMatrix(&out2);
	if (MatrixMath::IsEqual(&out1, &out2))
		std::cout << "The models are equal." << std::endl;

	return 0;
}
