#pragma once
#include "Layer.h"
class JsonIO
{
public:
	static JsonIO& GetInstance();
	JsonIO(const JsonIO&) = delete;
	void operator=(const JsonIO&) = delete;

	std::string SaveLayerToJSON(Layer* layer, Layer::LayerType type);
	

private:
	JsonIO() {}

	std::string SaveInputLayer(Layer* layer);
	std::string SaveFeedForwardLayer(Layer* layer);
	std::string SaveRecurrentLayer(Layer* layer);
	std::string SaveLSTM(Layer* layer);
};



