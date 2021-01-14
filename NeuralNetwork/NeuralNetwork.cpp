#include <iostream>
#include "Matrix.h"
#include "InputLayer.h"
#include "FeedForwardLayer.h"
#include "LSTM.h"
#include "ActivationFunctions.hpp"
#include "Constants.h"

#include "GradientDescent.h"
#include "LossFunctions.hpp"

#include "rapidjson/document.h"
#include "rapidjson/rapidjson.h"
#include "rapidjson/writer.h"
#include "rapidjson/ostreamwrapper.h"
#include <fstream>

int main()
{

	//creating matrix json
	/*rapidjson::Document document;
	document.SetObject();

	rapidjson::Value rows, cols;
	rows = 5;
	cols = 5;

	rapidjson::Value values(rapidjson::kArrayType);
	for (size_t i = 0; i < 25; i++)
		values.PushBack(i, document.GetAllocator());

	rapidjson::Value root(rapidjson::kObjectType);
	root.AddMember("rows", rows, document.GetAllocator());
	root.AddMember("cols", cols, document.GetAllocator());
	root.AddMember("values", values, document.GetAllocator());

	document.AddMember("root", root, document.GetAllocator());

	std::ofstream writer("output.json");
	rapidjson::OStreamWrapper osw(writer);
	rapidjson::Writer<rapidjson::OStreamWrapper> w(osw);
	document.Accept(w);
	writer.close();*/

	FeedForwardLayer layer(nullptr, 5);
	layer.SaveToJSON("layer.json");

	return 0;
}
