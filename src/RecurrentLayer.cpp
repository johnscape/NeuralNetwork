#include "RecurrentLayer.h"

#if USE_GPU
#include "MatrixGPUMath.cuh"
#endif

RecurrentLayer::RecurrentLayer(Layer* inputLayer, unsigned int size, unsigned int timeSteps) :
	Layer(inputLayer), TimeSteps(timeSteps), CurrentStep(0), Size(size),
	Weights(), Bias(1, size), InnerState(1, size), WeightError(), BiasError(1, size),
	RecursiveWeight(size, size), RecursiveWeightError(size, size)
{
	Weights.Reset(inputLayer->GetOutput().GetVectorSize(), size);
	Output.Reset(1, size);
	WeightError.Reset(inputLayer->GetOutput().GetVectorSize(), size);
	LayerError.Reset(1, inputLayer->GetOutput().GetVectorSize());
	function = &TanhFunction::GetInstance();

	Bias.FillWith(1);
	Weights.FillWithRandom();
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
	LayerInput->Compute();
	InnerState = LayerInput->GetOutput() * Weights;
	InnerState += Output * RecursiveWeight;
	InnerState += Bias;
	function->CalculateInto(InnerState, Output);
	if (TrainingMode)
	{
		TrainingStates.push_back(Matrix(InnerState));
		IncomingValues.push_back(Matrix((LayerInput->GetOutput())));
		if (TrainingStates.size() > TimeSteps)
		{
			TrainingStates.pop_front();
			IncomingValues.pop_front();
		}
	}
}

Matrix& RecurrentLayer::GetOutput()
{
	return Output;
}

Matrix& RecurrentLayer::ComputeAndGetOutput()
{
	Compute();
	return Output;
}

void RecurrentLayer::SetActivationFunction(ActivationFunction* func)
{
	if (function)
		delete function;
	function = func;
}

void RecurrentLayer::GetBackwardPass(const Matrix& error, bool recursive)
{
	Matrix derivate = function->CalculateDerivateMatrix(Output);
	LayerError.FillWith(0);
#if USE_GPU
	derivate->CopyFromGPU();
#endif // USE_GPU


	std::vector<Matrix> powers;
	for (unsigned int i = 0; i <= TimeSteps; i++)
	{
		if (!i)
			continue;
		powers.push_back(RecursiveWeight.Power(i));
	}

	for (unsigned int neuron = 0; neuron < Size; neuron++)
	{
		float delta = error.GetValue(neuron);
		delta *= derivate.GetValue(neuron);
		for (unsigned int time = 0; time < TimeSteps; time++)
		{
			if (TimeSteps - time >= TrainingStates.size())
				continue;
			for (unsigned int incoming = 0; incoming < LayerInput->GetOutput().GetVectorSize(); incoming++)
			{
				float wt = 0;
				if (time)
				{
					for (unsigned int recursive = 0; recursive < Size; recursive++)
						wt += error.GetValue(recursive) * derivate.GetValue(recursive) * powers[time].GetValue(neuron, recursive) * IncomingValues[TimeSteps - time].GetValue(incoming);
				}
				else
					wt = IncomingValues[IncomingValues.size() - 1].GetValue(incoming) * delta;
				WeightError.AdjustValue(incoming, neuron, wt);
				LayerError.AdjustValue(incoming, delta * Weights.GetValue(incoming, neuron));
			}
			for (unsigned int recursive = 0; recursive < Size; recursive++)
			{
				float wt = 0;
				if (time)
				{
					for (unsigned int r = 0; r < Size; r++)
						wt += error.GetValue(r) * derivate.GetValue(r) * powers[time].GetValue(neuron, r) * TrainingStates[TimeSteps - time].GetValue(recursive);
				}
				else
					wt = TrainingStates[TrainingStates.size() - 1].GetValue(recursive) * delta;

				RecursiveWeightError.AdjustValue(recursive, neuron, wt);
			}

		}

		BiasError.AdjustValue(neuron, delta);
	}

	powers.clear();

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
	if (!mode)
	{
		while (!TrainingStates.empty())
			TrainingStates.pop_front();
	}
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

