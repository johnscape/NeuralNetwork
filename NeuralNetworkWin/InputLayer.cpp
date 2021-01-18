#include "InputLayer.h"
#include "MatrixMath.h"
#include "LayerException.hpp"
#include "Constants.h"

InputLayer::InputLayer(unsigned int size) : Layer(), Size(size)
{
	LayerInput = nullptr;
	Output = new Matrix(1, size);
}

Layer* InputLayer::Clone()
{
	return new InputLayer(Size);
}

void InputLayer::Compute()
{
	return;
}

Matrix* InputLayer::ComputeAndGetOutput()
{
	return Output;
}

void InputLayer::SetInput(Matrix* input)
{
#if DEBUG
	if (!MatrixMath::SizeCheck(input, Output))
		return; //TODO: Throw error
#endif // DEBUG
	MatrixMath::Copy(input, Output);
#if USE_GPU
	Output->CopyToGPU();
#endif // USE_GPU

}

void InputLayer::GetBackwardPass(Matrix* error, bool recursive)
{
}

void InputLayer::LoadFromJSON(const char* data, bool isFile)
{
	rapidjson::Document document;
	if (!isFile)
		document.Parse(data);
	else
	{
		std::ifstream r(data);
		rapidjson::IStreamWrapper isw(r);
		document.ParseStream(isw);
	}

	rapidjson::Value val;
	val = document["layer"]["size"];
	Size = val.GetUint();
}

std::string InputLayer::SaveToJSON(const char* fileName)
{
	rapidjson::Document doc;
	doc.SetObject();

	rapidjson::Value input, id, type;
	input.SetUint(Size);
	id.SetUint(Id);
	type.SetUint(0);

	rapidjson::Value root(rapidjson::kObjectType);
	root.AddMember("id", id, doc.GetAllocator());
	root.AddMember("type", type, doc.GetAllocator());
	root.AddMember("size", input, doc.GetAllocator());

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

	return std::string(buffer.GetString());
}

