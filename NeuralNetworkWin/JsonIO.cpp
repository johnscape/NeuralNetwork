#include "JsonIO.h"
#include "InputLayer.h"


JsonIO& JsonIO::GetInstance()
{
	static JsonIO instance;
    return instance;
}

std::string JsonIO::SaveLayerToJSON(Layer* layer, Layer::LayerType type)
{
	switch (type)
	{
	case Layer::INPUT:
		return SaveInputLayer(layer);
		break;
	case Layer::FEEDFORWARD:
		return SaveFeedForwardLayer(layer);
		break;
	case Layer::RECURRENT:
		return SaveRecurrentLayer(layer);
		break;
	case Layer::LSTMLAYER:
		return SaveLSTM(layer);
		break;
	default:
		return "";
		break;
	}

	return "";
}

std::string JsonIO::SaveInputLayer(Layer* layer)
{
	InputLayer* current = dynamic_cast<InputLayer*>(layer);


	/*rapidjson::Document doc;
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

	rapidjson::StringBuffer buffer;
	rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
	doc.Accept(writer);

	return std::string(buffer.GetString());*/

	return "";
}

std::string JsonIO::SaveFeedForwardLayer(Layer* layer)
{
	return std::string();
}

std::string JsonIO::SaveRecurrentLayer(Layer* layer)
{
	return std::string();
}

std::string JsonIO::SaveLSTM(Layer* layer)
{
	return std::string();
}
