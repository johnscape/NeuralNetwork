#include "NeuralNetwork/Layers/RecurrentLayer.h"

#if USE_GPU
#include "MatrixGPUMath.cuh"
#endif

RecurrentLayer::RecurrentLayer(Layer* inputLayer, unsigned int size, unsigned int timeSteps) :
	Layer(inputLayer), TimeSteps(timeSteps), CurrentStep(0), Size(size),
	Weights(), Bias(1, size), InnerState(), WeightError(), BiasError(1, size),
	RecursiveWeight(size, size), RecursiveWeightError(size, size), RecursiveState(1, size)
{
	Weights.Reset(inputLayer->GetOutput().GetShapeAt(1), size);
	Output = Tensor({1, size}, nullptr);
	WeightError.Reset(inputLayer->GetOutput().GetShapeAt(1), size);
	function = &TanhFunction::GetInstance();

	Bias.FillWith(1);
	Weights.FillWithRandom();
	RecursiveWeight.FillWithRandom();
}

RecurrentLayer::~RecurrentLayer()
{
}

Layer* RecurrentLayer::Clone()
{
	RecurrentLayer* r = new RecurrentLayer(LayerInput, Size, TimeSteps);
	r->GetWeights().Copy(Weights);
	r->GetRecurrentWeights().Copy(RecursiveWeight);
	r->GetBias().Copy(Bias);
	return r;
}

void RecurrentLayer::Compute()
{
	IncomingValues = LayerInput->ComputeAndGetOutput();
	InnerState = IncomingValues * Weights;
	for (unsigned int mat = 0; mat < InnerState.GetMatrixCount(); ++mat)
	{
		for (unsigned int row = 0; row < InnerState.GetShapeAt(0); ++row)
		{
			Matrix rowMat = InnerState.GetRowMatrix(mat, row);
			rowMat += Bias;
			rowMat += RecursiveState * RecursiveWeight;
			RecursiveState = rowMat;
			for (unsigned int col = 0; col < InnerState.GetShapeAt(1); ++col)
			{
				unsigned int pos = mat * InnerState.GetShapeAt(0) * InnerState.GetShapeAt(1);
				pos += row * InnerState.GetShapeAt(1);
				pos += col;
				InnerState.SetValue(pos, RecursiveState.GetValue(col));
			}
		}
	}

	if (!Output.IsSameShape(InnerState))
		Output = Tensor(InnerState);

	function->CalculateInto(InnerState, Output);
}

Tensor& RecurrentLayer::GetOutput()
{
	return Output;
}

Tensor& RecurrentLayer::ComputeAndGetOutput()
{
	Compute();
	return Output;
}

void RecurrentLayer::SetActivationFunction(ActivationFunction* func)
{
	function = func;
}

void RecurrentLayer::GetBackwardPass(const Tensor& error, bool recursive)
{
	//TODO: Implement tensor elementwise multiply
	//LayerError = weight * (error .* derivate)
	TempMatrix errorMatrix = error.ToMatrixByRows();
	Tensor derivate = function->CalculateDerivateTensor(Output);
	LayerError = Tensor({(unsigned int)errorMatrix.GetRowCount(), LayerInput->OutputSize()}, nullptr);
#if USE_GPU
	derivate->CopyFromGPU();
#endif // USE_GPU

	TempMatrix states = InnerState.ToMatrixByRows();

	//If I call this function for once at every batch, this can stay, otherwise create a parameter
	std::vector<Matrix> powers;
	for (unsigned int i = 0; i < 3; ++i)
	{
		if (i == 0)
			powers.push_back(RecursiveWeight);
		else
			powers.push_back(RecursiveWeight.Power(i + 1));
	}

	for (unsigned int mat = 0; mat < error.GetMatrixCount(); ++mat)
	{
		for (unsigned int row = 0; row < error.GetShapeAt(0); ++row)
		{
			Matrix incoming = IncomingValues.GetRowMatrix(mat, row); //i_t
			incoming.Transpose();

			Matrix derivated = derivate.GetRowMatrix(mat, row); //o_t/s_t
			derivated.ElementwiseMultiply(error.GetRowMatrix(mat, row)); //E_t/s_t
			Matrix weightErr = incoming * derivated; //E_t/W

			for (int i = 0; i < TimeSteps; ++i) //Do I need to update for every timestep, or just for i=3?
			{
				if (i >= row)
					break;
				TempMatrix state = states.GetTempRowMatrix(row - i - 1);
				state.Transpose();
				if (i == 0)
					RecursiveWeightError += state * derivated;
				else
				{
					Matrix tmp = powers[i - 1] * state;
					tmp *= derivated;
					RecursiveWeightError += tmp;
				}

			}
			WeightError += weightErr;
			BiasError += derivated;
		}
	}

	if (recursive)
		LayerInput->GetBackwardPass(LayerError);
}

void RecurrentLayer::Train(Optimizer* optimizer)
{
	optimizer->ModifyWeights(Weights, WeightError);
	optimizer->ModifyWeights(RecursiveWeight, RecursiveWeightError);
	optimizer->ModifyWeights(Bias, BiasError);

	WeightError.FillWith(0);
	RecursiveWeightError.FillWith(0);
	BiasError.FillWith(0);

#if USE_GPU
	Weights->CopyToGPU();
	RecursiveWeight->CopyToGPU();
	Bias->CopyToGPU();
#endif // USE_GPU

}

void RecurrentLayer::SetTrainingMode(bool mode)
{
	TrainingMode = mode;
}

Matrix& RecurrentLayer::GetWeights()
{
	return Weights;
}

Matrix& RecurrentLayer::GetBias()
{
	return Bias;
}

Matrix& RecurrentLayer::GetRecurrentWeights()
{
	return RecursiveWeight;
}

void RecurrentLayer::LoadFromJSON(const char* data, bool isFile)
{
	/*rapidjson::Document document;
	if (!isFile)
		document.Parse(data);
	else
	{
		std::ifstream r(data);
		rapidjson::IStreamWrapper isw(r);
		document.ParseStream(isw);
	}
	rapidjson::Value val;


	unsigned int InputSize = 1;
	val = document["layer"]["size"];
	Size = val.GetUint();
	val = document["layer"]["inputSize"];

	unsigned int inputSize = val.GetUint();
	if (LayerInput)
		inputSize = LayerInput->GetOutput().GetVectorSize();
	Weights.Reset(inputSize, Size);
	Output.Reset(1, Size);
	Bias.Reset(1, Size);
	InnerState.Reset(1, Size);
	WeightError.Reset(inputSize, Size);
	LayerError.Reset(1, inputSize);
	BiasError.Reset(1, Size);
	RecursiveWeight.Reset(Size, Size);
	RecursiveWeightError.Reset(Size, Size);

	rapidjson::StringBuffer buffer;
	rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);

	document["layer"]["weights"].Accept(writer);
	Weights.LoadFromJSON(buffer.GetString());

	buffer.Clear();
	writer.Reset(buffer);

	document["layer"]["bias"].Accept(writer);
	Bias.LoadFromJSON(buffer.GetString());

	buffer.Clear();
	writer.Reset(buffer);

	document["layer"]["recurrent"].Accept(writer);
	RecursiveWeight.LoadFromJSON(buffer.GetString());*/

}

std::string RecurrentLayer::SaveToJSON(const char* fileName)
{
	/*rapidjson::Document doc;
	doc.SetObject();

	rapidjson::Value layerSize, id, type, inputSize;
	layerSize.SetUint(Size);
	id.SetUint(Id);
	type.SetUint(2);
	if (LayerInput)
		inputSize.SetUint(LayerInput->GetOutput().GetVectorSize());
	else
		inputSize.SetUint(1);

	rapidjson::Document weight, bias, recurrent;

	weight.Parse(Weights.SaveToJSON().c_str());
	bias.Parse(Bias.SaveToJSON().c_str());
	recurrent.Parse(RecursiveWeight.SaveToJSON().c_str());

	rapidjson::Value root(rapidjson::kObjectType);
	root.AddMember("id", id, doc.GetAllocator());
	root.AddMember("type", type, doc.GetAllocator());
	root.AddMember("size", layerSize, doc.GetAllocator());
	root.AddMember("inputSize", inputSize, doc.GetAllocator());
	root.AddMember("weights", weight, doc.GetAllocator());
	root.AddMember("bias", bias, doc.GetAllocator());
	root.AddMember("recurrent", recurrent, doc.GetAllocator());

	doc.AddMember("layer", root, doc.GetAllocator());

	if (fileName)
	{
		std::ofstream w(fileName);
		rapidjson::OStreamWrapper osw(w);
		rapidjson::Writer<rapidjson::OStreamWrapper> writer(osw);
		doc.Accept(writer);
		w.close();
	}

	rapidjson::StringBuffer buffer;
	rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
	doc.Accept(writer);

	return std::string(buffer.GetString());*/

	return "";
}

