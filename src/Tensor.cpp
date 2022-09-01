#include <utility>
#include <random>
#include "NeuralNetwork/Tensor.h"
#include "NeuralNetwork/TensorException.hpp"
#include "nmmintrin.h"
#include <numeric>

#include <fstream>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>

std::ostream& operator<<(std::ostream& os, const Tensor& tensor)
{
	unsigned int rowCount = tensor.GetShape()[0];
	unsigned int colCount = tensor.GetShape()[1];

	for (unsigned int i = 0; i < tensor.GetMatrixCount(); ++i)
	{
		for (unsigned int j = 0; j < rowCount; ++j)
		{
			for (unsigned int k = 0; k < colCount; ++k)
			{
				os << tensor.GetValue(k + colCount * j + colCount * rowCount * i) << '\t';
			}
			os << std::endl;
		}
		os << std::endl;
	}

	return os;
}

Tensor::Tensor() : Values(nullptr), ElementCount(0)
{
}

Tensor::Tensor(unsigned int* shape, unsigned int shapeCount, float* values) : ElementCount(1)
{
	if (shapeCount == 0)
		shapeCount = *(&shape + 1) - shape;

	for (unsigned int i = 0; i < shapeCount; i++)
	{
		Shape.push_back(shape[i]);
		ElementCount *= shape[i];
	}

	Values = new float[ElementCount];

	if (values != nullptr)
		std::copy(values, values + ElementCount, Values);
	else
		std::fill(Values, Values + ElementCount, 0);
}

Tensor::Tensor(std::vector<unsigned int> shape, float* values) : ElementCount(1)
{
	Shape = std::move(shape);
	for (unsigned int i : Shape)
		ElementCount *= i;

	Values = new float[ElementCount];

	if (values != nullptr)
		std::copy(values, values + ElementCount, Values);
	else
		std::fill(Values, Values + ElementCount, 0);
}

Tensor::Tensor(const Matrix& mat)
{
	ElementCount = mat.GetElementCount();
	Shape.push_back(mat.GetRowCount());
	Shape.push_back(mat.GetColumnCount());

	Values = new float[ElementCount];
	std::copy(mat.Values, mat.Values + ElementCount, Values);
}

Tensor::Tensor (Matrix&& mat) : Values(nullptr)
{
	Shape = {(unsigned int)mat.Rows, (unsigned int)mat.Columns};
	Values = mat.Values;
	mat.Values = nullptr;
}

Tensor::Tensor(const Tensor &other)
{
	ElementCount = other.ElementCount;
	Shape = other.Shape;
	Values = new float[ElementCount];
	std::copy(other.Values, other.Values + ElementCount, Values);
}


Tensor::Tensor(Tensor &&other) noexcept :
ElementCount(std::exchange(other.ElementCount, 0)), Values(std::exchange(other.Values,nullptr)),
Shape(std::move(other.Shape))
{
}


Tensor::~Tensor()
{
	if (Values)
		delete[] Values;
}

bool Tensor::IsSameShape(const Tensor& other) const
{
	if (Shape.size() != other.Shape.size())
		return false;
	return Shape == other.Shape;
}

bool Tensor::IsSameShape(const Matrix &other) const
{
	unsigned int rows = !Shape.empty() ? Shape[0] : 1;
	unsigned int cols = Shape.size() >= 2 ? Shape[1] : 1;


	return rows == other.GetRowCount() && cols == other.GetColumnCount();
}

void Tensor::Reshape(const std::vector<unsigned int>& newShape)
{
	unsigned int sum1 = Shape[0];
	unsigned int sum2 = newShape[0];

	for (unsigned int i = 1; i < Shape.size(); i++)
		sum1 *= Shape[i];
	
	for (unsigned int i = 1; i < newShape.size(); i++)
		sum2 *= newShape[i];

	if (sum1 != sum2)
		throw TensorShapeException();

	Shape = newShape;
}

void Tensor::Reshape(unsigned int* newDimensions, unsigned int dimensionCount)
{
	if (dimensionCount == 0)
		dimensionCount = *(&newDimensions + 1) - newDimensions;
	unsigned int sum1 = Shape[0];
	unsigned int sum2 = newDimensions[0];

	for (unsigned int i = 1; i < Shape.size(); i++)
		sum1 *= Shape[i];

	for (unsigned int i = 1; i < dimensionCount; i++)
		sum2 *= newDimensions[i];

	if (sum1 != sum2)
		throw TensorShapeException();

	Shape.clear();
	Shape = std::vector<unsigned int>(newDimensions, newDimensions + dimensionCount);
}

float Tensor::GetValue(unsigned int position) const
{
	if (position >= GetElementCount())
		throw TensorIndexException();
	return Values[position];
}

void Tensor::SetValue(unsigned int pos, float value)
{
	Values[pos] = value;
}

void Tensor::AdjustValue(unsigned int pos, float value)
{
	Values[pos] += value;
}

std::vector<unsigned int> Tensor::GetShape() const
{
	return Shape;
}

unsigned int Tensor::GetShapeAt(unsigned int i) const
{
	if (i >= Shape.size())
		return 1;
	return Shape[i];
}

Matrix Tensor::FirstMatrix() const
{
	if (Shape.size() >= 2)
	{
		Matrix mat(Shape[0], Shape[1]);
		std::copy(Values, Values + (Shape[0] * Shape[1]), mat.Values);
		return mat;
	}
	else if (Shape.size() == 1)
	{
		Matrix mat(Shape[0], 1);
		std::copy(Values, Values + Shape[0], mat.Values);
		return mat;
	}
	else
		return Matrix();
}

std::list<Matrix> Tensor::ToMatrixList() const
{
	std::list<Matrix> items;
	if (Shape.size() <= 2)
		items.push_back(FirstMatrix());
	else
	{
		unsigned int itemCount = Shape[2];
		if (Shape.size() >= 3)
			for (size_t i = 3; i < Shape.size(); i++)
				itemCount *= Shape[i];
		unsigned int elementCount = Shape[0] * Shape[1];
		
		for (size_t i = 0; i < itemCount; i++)
		{
			Matrix m(Shape[0], Shape[1]);
			std::copy(Values + i * elementCount, Values + (i + 1) * elementCount, m.Values);
			items.push_back(m);
		}
	}
	return items;
}

void Tensor::GetNthMatrix(unsigned int n, Matrix* mat)
{
	if (!mat)
		mat = new Matrix(Shape[0], Shape[1]);
	else if (mat->GetRowCount() != Shape[0] || mat->GetColumnCount() != Shape[1])
		throw TensorShapeException();

	if (n >= GetMatrixCount())
		mat->FillWith(0);
	else
		std::copy(Values + n * (Shape[0] * Shape[1]), Values + (n + 1) * (Shape[0] * Shape[1]), mat->Values);
}

TempMatrix Tensor::GetNthTempMatrix(unsigned int n) const
{
	if (n >= GetMatrixCount())
		throw TensorIndexException();
	unsigned int matSize = Shape[0];
	unsigned int rows = Shape[0];
	unsigned int cols = Shape[1];
	if (Shape.size() >= 2)
	{
		matSize *= Shape[1];
		cols = Shape[1];
	}

	return TempMatrix(rows, cols, Values + n * matSize);
}

Matrix Tensor::GetRowMatrix(unsigned int matrix, unsigned int row) const
{
	return Matrix(1, Shape[1], Values + matrix * Shape[0] * Shape[1] + row * Shape[1]);
}

TempMatrix Tensor::ToMatrixByRows() const
{
	return TempMatrix(Shape[0] * GetMatrixCount(), Shape[1], Values);
}

void Tensor::LoadMatrix(unsigned int n, Matrix* mat)
{
	if (n >= GetMatrixCount())
		return;
	if (mat->GetRowCount() != Shape[0] || mat->GetColumnCount() != Shape[1])
		throw TensorShapeException();

	std::copy(mat->Values, mat->Values + mat->GetElementCount(), Values + n * Shape[0] * Shape[1]);
}

unsigned int Tensor::GetElementCount() const
{
	unsigned int sum = 1;
	for (unsigned int i = 0; i < Shape.size(); ++i)
		sum *= Shape[i];
	return sum;
}

float Tensor::Sum() const
{
	float sum = 0;
	for (int i = 0; i < GetElementCount(); ++i)
		sum += Values[i];
	return sum;
}

void Tensor::Squeeze()
{
	std::vector<unsigned int> newShape;
	for (unsigned int s : Shape)
		if (s != 1)
			newShape.push_back(s);
	if (newShape.empty())
		newShape.push_back(1);
	Shape = newShape;
}

void Tensor::FillWith(float value)
{
	std::fill(Values, Values + GetElementCount(), value);
}

void Tensor::FillWithRandom(float min, float max)
{
	if (min > max)
	{
		max += min;
		min = max - min;
		max -= min;
	}
	static std::random_device device;
	static std::mt19937 engine;
	engine.seed(device());
	std::uniform_real_distribution<> dist(min, max);

	for (size_t i = 0; i < GetElementCount(); i++)
		Values[i] = static_cast<float>(dist(engine));
}

void Tensor::LoadFromJSON(const char *data, bool isFile)
{
	rapidjson::Document document;
	if (!isFile)
		document.Parse(data);
	else
	{
		std::ifstream reader(data);
		rapidjson::IStreamWrapper jsonReader(reader);
		document.ParseStream(jsonReader);
	}
	rapidjson::Value value;
	value = document["tensor"];
	LoadFromJSON(value);

}
void Tensor::LoadFromJSON(rapidjson::Value& jsonValue)
{
	if (Values)
		delete[] Values;
	Shape.clear();

	if (jsonValue.HasMember("tensor"))
		jsonValue = jsonValue["tensor"];

	rapidjson::Value value;
	value = jsonValue["shape"];
	for (rapidjson::Value::ConstValueIterator itr = value.Begin(); itr != value.End(); itr++)
		Shape.push_back(itr->GetUint64());
	value = jsonValue["values"];
	Values = new float[GetElementCount()];
	unsigned int counter = 0;
	for (rapidjson::Value::ConstValueIterator itr = value.Begin(); itr != value.End(); itr++)
	{
		Values[counter] = itr->GetFloat();
		counter++;
	}
}


std::string Tensor::SaveToJSON(const char *fileName) const
{
	rapidjson::Document document;
	document.SetObject();
	rapidjson::Value tensor = SaveToJSONObject(document);

	document.AddMember("tensor", tensor, document.GetAllocator());

	if (fileName) //save json to file
	{
		std::ofstream writer(fileName);
		rapidjson::OStreamWrapper wrapper(writer);
		rapidjson::Writer<rapidjson::OStreamWrapper> jsonWriter(wrapper);
		document.Accept(jsonWriter);
		writer.close();
	}

	rapidjson::StringBuffer buffer;
	rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
	document.Accept(writer);

	return std::string(buffer.GetString());
}

rapidjson::Value Tensor::SaveToJSONObject(rapidjson::Document& document) const
{
	rapidjson::Value shape, values;
	rapidjson::Value tensor(rapidjson::kObjectType);

	shape.SetArray();
	values.SetArray();
	for (unsigned int s = 0; s < Shape.size(); ++s)
		shape.PushBack(Shape[s], document.GetAllocator());
	for (unsigned int v = 0; v < GetElementCount(); v++)
		values.PushBack(Values[v], document.GetAllocator());

	tensor.AddMember("shape", shape, document.GetAllocator());
	tensor.AddMember("values", values, document.GetAllocator());

	return tensor;
}

unsigned int Tensor::GetMatrixCount() const
{
	if (Shape.size() <= 2)
		return 1;

	unsigned int count = 1;
	for (int i = 2; i < Shape.size(); ++i)
		count *= Shape[i];
	return count;
}

unsigned int Tensor::CoordinateToPos(unsigned int* coord) const
{
	//in a perfect world, the length of the coord will be always equal (or larger) than the number of the dimensions
	//At the altar of speed, I have sacrificed safety
	if (GetElementCount() == 0)
		return 0;
	if (Shape.size() == 1)
		return coord[0];
	if (Shape.size() == 2)
		return coord[0] * Shape[1] + coord[1];

	unsigned int pos = coord[0] * Shape[1] + coord[1];
	unsigned int pastSize = Shape[0] * Shape[1];

	for (int i = 2; i < Shape.size(); ++i)
	{
		pos += pastSize * coord[i];
		pastSize *= Shape[i];
	}

	return pos;
}

unsigned int Tensor::CoordinateToPos(std::vector<unsigned int> coord) const
{
	return CoordinateToPos(&coord[0]);
}

Tensor Tensor::operator+(const Matrix &other) const
{
	if (!IsSameShape(other))
		throw TensorShapeException();

	Tensor t(*this);

	unsigned matrixCount = other.GetElementCount();

	unsigned int n4 = matrixCount - (matrixCount % 4);

	for (int mat = 0; mat < GetMatrixCount(); ++mat)
	{
		for (int val = 0; val < n4; val += 4)
		{
			float tensorVals[4] = {
					Values[mat * matrixCount + 	val		],
					Values[mat * matrixCount + 	val + 1	],
					Values[mat * matrixCount + 	val + 2	],
					Values[mat * matrixCount + 	val + 3	]
			};

			float matrixVals[4] = {
					other.Values[val],
					other.Values[val + 1],
					other.Values[val + 2],
					other.Values[val + 3],
			};

			float result[4];

			__m128 tensorVec = _mm_load_ps(tensorVals);
			__m128 matrixVec = _mm_load_ps(matrixVals);
			__m128 resultVec = _mm_add_ps(tensorVec, matrixVec);
			_mm_store_ps(result, resultVec);

			std::copy(result, result + 4, t.Values + val + mat * matrixCount);
		}
		for (int val = n4; val < matrixCount; ++val)
		{
			t.Values[val + mat * matrixCount] = other.GetValue(val) + GetValue(val + mat * matrixCount);
		}
	}

	return t;
}

Tensor Tensor::operator-(const Matrix &other) const
{
	if (!IsSameShape(other))
		throw TensorShapeException();

	Tensor t(*this);

	unsigned matrixCount = other.GetElementCount();

	unsigned int n4 = matrixCount - (matrixCount % 4);

	for (int mat = 0; mat < GetMatrixCount(); ++mat)
	{
		for (int val = 0; val < n4; val += 4)
		{
			float tensorVals[4] = {
					Values[mat * matrixCount + 	val		],
					Values[mat * matrixCount + 	val + 1	],
					Values[mat * matrixCount + 	val + 2	],
					Values[mat * matrixCount + 	val + 3	]
			};

			float matrixVals[4] = {
					other.Values[val],
					other.Values[val + 1],
					other.Values[val + 2],
					other.Values[val + 3],
					};

			float result[4];

			__m128 tensorVec = _mm_load_ps(tensorVals);
			__m128 matrixVec = _mm_load_ps(matrixVals);
			__m128 resultVec = _mm_sub_ps(tensorVec, matrixVec);
			_mm_store_ps(result, resultVec);

			std::copy(result, result + 4, t.Values + val + mat * matrixCount);
		}
		for (int val = n4; val < matrixCount; ++val)
		{
			t.SetValue(val + mat * matrixCount, GetValue(val + mat * matrixCount) - other.GetValue(val));
		}
	}

	return t;
}

Tensor Tensor::operator*(const Matrix &other) const
{
	unsigned int rows = !Shape.empty() ? Shape[0] : 1;
	unsigned int cols = Shape.size() > 1 ? Shape[1] : 1;
	unsigned int matrixSize = rows * cols;

	if (cols != other.Rows)
		throw TensorShapeException();

	std::vector<unsigned int> newSize = {rows, static_cast<unsigned int>(other.Columns)};
	for (int i = 2; i < Shape.size(); ++i)
		newSize.push_back(Shape[i]);

	Tensor result(newSize);

	__m128 fastCol, fastRow, fastRes;

	for (int m = 0; m < GetMatrixCount(); ++m)
	{
		unsigned int matrixStart = m * rows * other.GetColumnCount();

		for (unsigned int col_count = 0; col_count < other.GetColumnCount(); col_count++)
		{
			for (unsigned int row_count = 0; row_count < other.GetRowCount(); row_count += 4)
			{
				float col[4] = {
						row_count < other.GetRowCount() ? other.GetValue(row_count, col_count) : 0,
						row_count + 1 < other.GetRowCount() ? other.GetValue(row_count + 1, col_count) : 0,
						row_count + 2 < other.GetRowCount() ? other.GetValue(row_count + 2, col_count) : 0,
						row_count + 3 < other.GetRowCount() ? other.GetValue(row_count + 3, col_count) : 0,
				};

				fastCol = _mm_load_ps(col);

				for (unsigned int i = 0; i < rows; ++i)
				{
					float row[4] = {
							row_count < cols ? Values[m * matrixSize + i * cols + row_count] : 0,
							row_count + 1 < cols ? Values[m * matrixSize + i * cols + row_count + 1] : 0,
							row_count + 2 < cols ? Values[m * matrixSize + i * cols + row_count + 2] : 0,
							row_count + 3 < cols ? Values[m * matrixSize + i * cols + row_count + 3] : 0,
					};

					fastRow = _mm_load_ps(row);
					fastRes = _mm_mul_ps(fastRow, fastCol);
					fastRes = _mm_hadd_ps(fastRes, fastRes);
					fastRes = _mm_hadd_ps(fastRes, fastRes);
					float fRes = _mm_cvtss_f32(fastRes);

					result.Values[matrixStart + i * other.GetColumnCount() + col_count] += fRes;
				}
			}
		}
	}

	return result;
}

Tensor &Tensor::operator+=(const Matrix &other)
{
	if (!IsSameShape(other))
	{
		if (Shape[1] == other.GetColumnCount())
			return RowBasedAddition(other, true);
		throw TensorShapeException();
	}

	Tensor t(*this);

	unsigned matrixCount = other.GetElementCount();

	unsigned int n4 = matrixCount - (matrixCount % 4);

	for (int mat = 0; mat < GetMatrixCount(); ++mat)
	{
		for (int val = 0; val < n4; val += 4)
		{
			float tensorVals[4] = {
					Values[mat * matrixCount + 	val		],
					Values[mat * matrixCount + 	val + 1	],
					Values[mat * matrixCount + 	val + 2	],
					Values[mat * matrixCount + 	val + 3	]
			};

			float matrixVals[4] = {
					other.Values[val],
					other.Values[val + 1],
					other.Values[val + 2],
					other.Values[val + 3],
			};

			float result[4];

			__m128 tensorVec = _mm_load_ps(tensorVals);
			__m128 matrixVec = _mm_load_ps(matrixVals);
			__m128 resultVec = _mm_add_ps(tensorVec, matrixVec);
			_mm_store_ps(result, resultVec);

			std::copy(result, result + 4, t.Values + val + mat * matrixCount);
		}
		for (int val = n4; val < matrixCount; ++val)
		{
			t.SetValue(val + mat * matrixCount, other.GetValue(val) + GetValue(val + mat * matrixCount));
		}
	}

	ReloadFromOther(t);

	return *this;
}

Tensor &Tensor::operator-=(const Matrix &other)
{
	if (!IsSameShape(other))
		throw TensorShapeException();

	Tensor t(*this);

	unsigned matrixCount = other.GetElementCount();

	unsigned int n4 = matrixCount - (matrixCount % 4);

	for (int mat = 0; mat < GetMatrixCount(); ++mat)
	{
		for (int val = 0; val < n4; val += 4)
		{
			float tensorVals[4] = {
					Values[mat * matrixCount + 	val		],
					Values[mat * matrixCount + 	val + 1	],
					Values[mat * matrixCount + 	val + 2	],
					Values[mat * matrixCount + 	val + 3	]
			};

			float matrixVals[4] = {
					other.Values[val],
					other.Values[val + 1],
					other.Values[val + 2],
					other.Values[val + 3],
			};

			float result[4];

			__m128 tensorVec = _mm_load_ps(tensorVals);
			__m128 matrixVec = _mm_load_ps(matrixVals);
			__m128 resultVec = _mm_sub_ps(tensorVec, matrixVec);
			_mm_store_ps(result, resultVec);

			std::copy(result, result + 4, t.Values + val + mat * matrixCount);
		}
		for (int val = n4; val < matrixCount; ++val)
		{
			t.SetValue(val + mat * matrixCount, GetValue(val + mat * matrixCount) - other.GetValue(val));
		}
	}

	ReloadFromOther(t);
	return *this;
}

Tensor &Tensor::operator*=(const Matrix &other)
{
	unsigned int rows = !Shape.empty() ? Shape[0] : 1;
	unsigned int cols = Shape.size() > 1 ? Shape[1] : 1;
	unsigned int matrixSize = rows * cols;

	if (cols != other.Rows)
		throw TensorShapeException();

	std::vector<unsigned int> newSize = {rows, static_cast<unsigned int>(other.Columns)};
	for (int i = 2; i < Shape.size(); ++i)
		newSize.push_back(Shape[i]);

	Tensor result(newSize);

	__m128 fastCol, fastRow, fastRes;

	for (int m = 0; m < GetMatrixCount(); ++m)
	{
		unsigned int matrixStart = m * rows * other.GetColumnCount();

		for (unsigned int col_count = 0; col_count < other.GetColumnCount(); col_count++)
		{
			for (unsigned int row_count = 0; row_count < other.GetRowCount(); row_count += 4)
			{
				float col[4] = {
						row_count < other.GetRowCount() ? other.GetValue(row_count, col_count) : 0,
						row_count + 1 < other.GetRowCount() ? other.GetValue(row_count + 1, col_count) : 0,
						row_count + 2 < other.GetRowCount() ? other.GetValue(row_count + 2, col_count) : 0,
						row_count + 3 < other.GetRowCount() ? other.GetValue(row_count + 3, col_count) : 0,
				};

				fastCol = _mm_load_ps(col);

				for (unsigned int i = 0; i < rows; ++i)
				{
					float row[4] = {
							row_count < cols ? Values[m * matrixSize + i * cols + row_count] : 0,
							row_count + 1 < cols ? Values[m * matrixSize + i * cols + row_count + 1] : 0,
							row_count + 2 < cols ? Values[m * matrixSize + i * cols + row_count + 2] : 0,
							row_count + 3 < cols ? Values[m * matrixSize + i * cols + row_count + 3] : 0,
					};

					fastRow = _mm_load_ps(row);
					fastRes = _mm_mul_ps(fastRow, fastCol);
					fastRes = _mm_hadd_ps(fastRes, fastRes);
					fastRes = _mm_hadd_ps(fastRes, fastRes);
					float fRes = _mm_cvtss_f32(fastRes);

					result.Values[matrixStart + i * other.GetColumnCount() + col_count] += fRes;
				}
			}
		}
	}

	ReloadFromOther(result);
	return *this;
}

bool Tensor::operator==(const Matrix& other) const
{
	if (Shape.size() > 2)
		return false;
	if (Shape[0] != other.GetRowCount() || Shape[1] != other.GetColumnCount())
		return false;
	for (int i = 0; i < other.GetElementCount(); ++i)
	{
		if (Values[i] != other.Values[i])
			return false;
	}

	return true;
}

bool Tensor::operator!=(const Matrix& other) const
{
	if (Shape.size() > 2)
		return true;
	if (Shape[0] != other.GetRowCount() || Shape[1] != other.GetColumnCount())
		return true;
	for (int i = 0; i < other.GetElementCount(); ++i)
	{
		if (Values[i] != other.Values[i])
			return true;
	}

	return false;
}

Tensor Tensor::operator+(const Tensor &other) const
{
	if (!IsSameShape(other))
		throw TensorShapeException();

	Tensor result(Shape);

	unsigned int rows = Shape.size() > 0 ? Shape[0] : 1;
	unsigned int cols = Shape.size() > 1 ? Shape[1] : 1;
	unsigned int matrixCount = rows * cols;

	unsigned int n4 = matrixCount - (matrixCount % 4);

	for (unsigned int m = 0; m < GetMatrixCount(); ++m)
	{
		for (int val = 0; val < n4; val += 4)
		{
			float tensorVals[4] = {
					Values[m * matrixCount + 	val		],
					Values[m * matrixCount + 	val + 1	],
					Values[m * matrixCount + 	val + 2	],
					Values[m * matrixCount + 	val + 3	]
			};

			float otherVals[4] = {
					other.Values[m * matrixCount + 	val		],
					other.Values[m * matrixCount + 	val + 1	],
					other.Values[m * matrixCount + 	val + 2	],
					other.Values[m * matrixCount + 	val + 3	],
			};

			float res[4];

			__m128 tensorVec = _mm_load_ps(tensorVals);
			__m128 matrixVec = _mm_load_ps(otherVals);
			__m128 resultVec = _mm_add_ps(tensorVec, matrixVec);
			_mm_store_ps(res, resultVec);

			std::copy(res, res + 4, result.Values + val + m * matrixCount);
		}
		for (unsigned int val = n4; val < matrixCount; ++val)
		{
			unsigned int pos = val + m * matrixCount;
			float res = other.GetValue(pos) + GetValue(pos);
			result.SetValue(pos, res);
		}
	}

	return result;
}

Tensor Tensor::operator-(const Tensor &other) const
{
	if (!IsSameShape(other))
		throw TensorShapeException();

	Tensor result(Shape);

	unsigned int rows = Shape.size() > 0 ? Shape[0] : 1;
	unsigned int cols = Shape.size() > 1 ? Shape[1] : 1;
	unsigned int matrixCount = rows * cols;

	unsigned int n4 = matrixCount - (matrixCount % 4);

	for (unsigned int m = 0; m < GetMatrixCount(); ++m)
	{
		for (int val = 0; val < n4; val += 4)
		{
			float tensorVals[4] = {
					Values[m * matrixCount + 	val		],
					Values[m * matrixCount + 	val + 1	],
					Values[m * matrixCount + 	val + 2	],
					Values[m * matrixCount + 	val + 3	]
			};

			float otherVals[4] = {
					other.Values[m * matrixCount + 	val		],
					other.Values[m * matrixCount + 	val + 1	],
					other.Values[m * matrixCount + 	val + 2	],
					other.Values[m * matrixCount + 	val + 3	],
			};

			float res[4];

			__m128 tensorVec = _mm_load_ps(tensorVals);
			__m128 matrixVec = _mm_load_ps(otherVals);
			__m128 resultVec = _mm_sub_ps(tensorVec, matrixVec);
			_mm_store_ps(res, resultVec);

			std::copy(res, res + 4, result.Values + val + m * matrixCount);
		}
		for (unsigned int val = n4; val < matrixCount; ++val)
		{
			unsigned int pos = val + m * matrixCount;
			float res = GetValue(pos) - other.GetValue(pos);
			result.SetValue(pos, res);
		}
	}

	return result;
}

Tensor Tensor::operator*(const Tensor &other) const
{
	unsigned int thisRows = !Shape.empty() ? Shape[0] : 1;
	unsigned int thisCols = Shape.size() > 1 ? Shape[1] : 1;

	unsigned int otherRows = !other.Shape.empty() ? other.Shape[0] : 1;
	unsigned int otherCols = other.Shape.size() > 1 ? other.Shape[1] : 1;

	unsigned int thisMatSize = thisRows * thisCols;
	unsigned int otherMatSize = otherRows * otherCols;

	if (thisCols != otherRows || Shape.size() != other.Shape.size())
		throw TensorShapeException();

	for (int i = 2; i < Shape.size(); ++i)
		if (Shape[i] != other.Shape[i])
			throw TensorShapeException();

	std::vector<unsigned int> newSize = {thisRows, static_cast<unsigned int>(otherCols)};
	for (int i = 2; i < Shape.size(); ++i)
		newSize.push_back(Shape[i]);

	Tensor result(newSize);

	__m128 fastCol, fastRow, fastRes;

	for (int m = 0; m < GetMatrixCount(); ++m)
	{
		unsigned int matrixStart = m * thisRows * otherCols;

		for (unsigned int col_count = 0; col_count < otherCols; col_count++)
		{
			for (unsigned int row_count = 0; row_count < otherRows; row_count += 4)
			{
				float col[4] = {
						row_count < otherRows ? other.Values[m * otherMatSize + col_count + row_count * otherCols] : 0, //other.GetValue(row_count, col_count) : 0,
						row_count + 1 < otherRows ? other.Values[m * otherMatSize + col_count + (row_count + 1) * otherCols] : 0,
						row_count + 2 < otherRows ? other.Values[m * otherMatSize + col_count + (row_count + 2) * otherCols] : 0,
						row_count + 3 < otherRows ? other.Values[m * otherMatSize + col_count + (row_count + 3) * otherCols] : 0,
				};

				fastCol = _mm_load_ps(col);

				for (unsigned int i = 0; i < thisRows; ++i)
				{
					float row[4] = {
							row_count < thisCols ? Values[m * thisMatSize + i * thisCols + row_count] : 0,
							row_count + 1 < thisCols ? Values[m * thisMatSize + i * thisCols + row_count + 1] : 0,
							row_count + 2 < thisCols ? Values[m * thisMatSize + i * thisCols + row_count + 2] : 0,
							row_count + 3 < thisCols ? Values[m * thisMatSize + i * thisCols + row_count + 3] : 0,
					};

					fastRow = _mm_load_ps(row);
					fastRes = _mm_mul_ps(fastRow, fastCol);
					fastRes = _mm_hadd_ps(fastRes, fastRes);
					fastRes = _mm_hadd_ps(fastRes, fastRes);
					float fRes = _mm_cvtss_f32(fastRes);

					result.Values[matrixStart + i * otherCols + col_count] += fRes;
				}
			}
		}
	}

	return result;
}

Tensor &Tensor::operator+=(const Tensor &other)
{
	if (!IsSameShape(other))
		throw TensorShapeException();

	Tensor result(Shape);

	unsigned int rows = Shape.size() > 0 ? Shape[0] : 1;
	unsigned int cols = Shape.size() > 1 ? Shape[1] : 1;
	unsigned int matrixCount = rows * cols;

	unsigned int n4 = matrixCount - (matrixCount % 4);

	for (unsigned int m = 0; m < GetMatrixCount(); ++m)
	{
		for (int val = 0; val < n4; val += 4)
		{
			float tensorVals[4] = {
					Values[m * matrixCount + 	val		],
					Values[m * matrixCount + 	val + 1	],
					Values[m * matrixCount + 	val + 2	],
					Values[m * matrixCount + 	val + 3	]
			};

			float otherVals[4] = {
					other.Values[m * matrixCount + 	val		],
					other.Values[m * matrixCount + 	val + 1	],
					other.Values[m * matrixCount + 	val + 2	],
					other.Values[m * matrixCount + 	val + 3	],
			};

			float res[4];

			__m128 tensorVec = _mm_load_ps(tensorVals);
			__m128 matrixVec = _mm_load_ps(otherVals);
			__m128 resultVec = _mm_add_ps(tensorVec, matrixVec);
			_mm_store_ps(res, resultVec);

			std::copy(res, res + 4, result.Values + val + m * matrixCount);
		}
		for (unsigned int val = n4; val < matrixCount; ++val)
		{
			unsigned int pos = val + m * matrixCount;
			float res = other.GetValue(pos) + GetValue(pos);
			result.SetValue(pos, res);
		}
	}

	ReloadFromOther(result);
	return *this;
}

Tensor &Tensor::operator-=(const Tensor &other)
{
	if (!IsSameShape(other))
		throw TensorShapeException();

	Tensor result(Shape);

	unsigned int rows = Shape.size() > 0 ? Shape[0] : 1;
	unsigned int cols = Shape.size() > 1 ? Shape[1] : 1;
	unsigned int matrixCount = rows * cols;

	unsigned int n4 = matrixCount - (matrixCount % 4);

	for (unsigned int m = 0; m < GetMatrixCount(); ++m)
	{
		for (int val = 0; val < n4; val += 4)
		{
			float tensorVals[4] = {
					Values[m * matrixCount + 	val		],
					Values[m * matrixCount + 	val + 1	],
					Values[m * matrixCount + 	val + 2	],
					Values[m * matrixCount + 	val + 3	]
			};

			float otherVals[4] = {
					other.Values[m * matrixCount + 	val		],
					other.Values[m * matrixCount + 	val + 1	],
					other.Values[m * matrixCount + 	val + 2	],
					other.Values[m * matrixCount + 	val + 3	],
			};

			float res[4];

			__m128 tensorVec = _mm_load_ps(tensorVals);
			__m128 matrixVec = _mm_load_ps(otherVals);
			__m128 resultVec = _mm_sub_ps(tensorVec, matrixVec);
			_mm_store_ps(res, resultVec);

			std::copy(res, res + 4, result.Values + val + m * matrixCount);
		}
		for (unsigned int val = n4; val < matrixCount; ++val)
		{
			unsigned int pos = val + m * matrixCount;
			float res = GetValue(pos) - other.GetValue(pos);
			result.SetValue(pos, res);
		}
	}

	ReloadFromOther(result);
	return *this;
}

Tensor &Tensor::operator*=(const Tensor &other)
{
	unsigned int thisRows = !Shape.empty() ? Shape[0] : 1;
	unsigned int thisCols = Shape.size() > 1 ? Shape[1] : 1;

	unsigned int otherRows = !other.Shape.empty() ? other.Shape[0] : 1;
	unsigned int otherCols = other.Shape.size() > 1 ? other.Shape[1] : 1;

	unsigned int thisMatSize = thisRows * thisCols;
	unsigned int otherMatSize = otherRows * otherCols;

	if (thisCols != otherRows || Shape.size() != other.Shape.size())
		throw TensorShapeException();

	for (int i = 2; i < Shape.size(); ++i)
		if (Shape[i] != other.Shape[i])
			throw TensorShapeException();

	std::vector<unsigned int> newSize = {thisRows, static_cast<unsigned int>(otherCols)};
	for (int i = 2; i < Shape.size(); ++i)
		newSize.push_back(Shape[i]);

	Tensor result(newSize);

	__m128 fastCol, fastRow, fastRes;

	for (int m = 0; m < GetMatrixCount(); ++m)
	{
		unsigned int matrixStart = m * thisRows * otherCols;

		for (unsigned int col_count = 0; col_count < otherCols; col_count++)
		{
			for (unsigned int row_count = 0; row_count < otherRows; row_count += 4)
			{
				float col[4] = {
						row_count < otherRows ? other.Values[m * otherMatSize + col_count + row_count * otherCols] : 0, //other.GetValue(row_count, col_count) : 0,
						row_count + 1 < otherRows ? other.Values[m * otherMatSize + col_count + (row_count + 1) * otherCols] : 0,
						row_count + 2 < otherRows ? other.Values[m * otherMatSize + col_count + (row_count + 2) * otherCols] : 0,
						row_count + 3 < otherRows ? other.Values[m * otherMatSize + col_count + (row_count + 3) * otherCols] : 0,
				};

				fastCol = _mm_load_ps(col);

				for (unsigned int i = 0; i < thisRows; ++i)
				{
					float row[4] = {
							row_count < thisCols ? Values[m * thisMatSize + i * thisCols + row_count] : 0,
							row_count + 1 < thisCols ? Values[m * thisMatSize + i * thisCols + row_count + 1] : 0,
							row_count + 2 < thisCols ? Values[m * thisMatSize + i * thisCols + row_count + 2] : 0,
							row_count + 3 < thisCols ? Values[m * thisMatSize + i * thisCols + row_count + 3] : 0,
					};

					fastRow = _mm_load_ps(row);
					fastRes = _mm_mul_ps(fastRow, fastCol);
					fastRes = _mm_hadd_ps(fastRes, fastRes);
					fastRes = _mm_hadd_ps(fastRes, fastRes);
					float fRes = _mm_cvtss_f32(fastRes);

					result.Values[matrixStart + i * otherCols + col_count] += fRes;
				}
			}
		}
	}

	ReloadFromOther(result);
	return *this;
}

Tensor &Tensor::operator=(const Tensor &other)
{
	ReloadFromOther(other);
	return *this;
}

Tensor &Tensor::operator=(Tensor &&other) noexcept
{
	Shape = std::move(other.Shape);
	std::swap(Values, other.Values);
	ElementCount = other.ElementCount;
	return *this;
}

bool Tensor::operator==(const Tensor &other) const
{
	if (!IsSameShape(other))
		return false;
	for (unsigned int i = 0; i < ElementCount; i++)
		if (Values[i] != other.Values[i])
			return false;
	return true;
}

bool Tensor::operator!=(const Tensor &other) const
{
	if (!IsSameShape(other))
		return true;
	for (unsigned int i = 0; i < ElementCount; i++)
		if (Values[i] != other.Values[i])
			return true;
	return false;
}

void Tensor::ReloadFromOther(const Tensor &other)
{
	delete [] Values;
	Shape = other.Shape;
	if (Shape.empty())
	{
		ElementCount = 0;
		Values = nullptr;
	}
	else
	{
		ElementCount = 1;
		for (int i = 0; i < Shape.size(); ++i)
		{
			ElementCount *= Shape[i];
		}

		Values = new float[ElementCount];
		std::copy(other.Values, other.Values + other.GetElementCount(), Values);
	}

}

void Tensor::Copy(const Tensor& other)
{
#if DEBUG
	if (!IsSameShape(other))
		throw TensorShapeException();
#endif
	std::copy(other.Values, other.Values + GetElementCount(), Values);
}

void Tensor::ReloadFromOther(const Matrix &other)
{
	delete [] Values;
	Shape = {static_cast<unsigned int>(other.Rows), static_cast<unsigned int>(other.Columns)};

	ElementCount = Shape[0] * Shape[1];
	Values = new float[ElementCount];
	std::copy(other.Values, other.Values + other.GetElementCount(), Values);
}

Tensor& Tensor::RowBasedAddition(const Matrix &mat, bool local)
{
	Matrix newMat(Shape[0], Shape[1]);
	for (unsigned int i = 0; i < Shape[0]; ++i)
		std::copy(mat.Values, mat.Values + Shape[1], newMat.Values + i * Shape[1]);

	Tensor result = (*this) + newMat;
	if (local)
	{
		ReloadFromOther(result);
		return *this;
	}
	else
		return result;
}
