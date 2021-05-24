#include "Matrix.h"
#include "Constants.h"
#include "MatrixException.hpp"
#include <fstream>
#include <string>

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "rapidjson/ostreamwrapper.h"
#include "rapidjson/istreamwrapper.h"

#if USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif // USE_GPU

#include "MatrixMath.h"

//TODO: Throw error at DEBUG errors

Matrix::Matrix() : GPUValues(nullptr)
{
	Columns = 1;
	Rows = 1;
	Values = new float[1];
	Values[0] = 0;

#if USE_GPU
	cudaMalloc((void**)&GPUValues, sizeof(float));
	CopyToGPU();
#endif // USE_GPU

}

Matrix::Matrix(size_t rows, size_t columns, float* elements) : GPUValues(nullptr)
{
	Columns = columns;
	Rows = rows;
	size_t count = GetElementCount();
	Values = new float[count];

	if (elements)
	{
		/*for (size_t i = 0; i < MaxValue; i++)
			Values[i] = elements[i];*/
		std::copy(elements, elements + count, Values);
	}
	else
	{
		/*for (size_t i = 0; i < MaxValue; i++)
			Values[i] = 0;*/
		std::fill(Values, Values + count, 0);
	}

#if USE_GPU
	cudaMalloc((void**)&GPUValues, sizeof(float) * MaxValue);
	CopyToGPU();
#endif // USE_GPU
}

Matrix::Matrix(const Matrix& c) : GPUValues(nullptr)
{
	Columns = c.GetColumnCount();
	Rows = c.GetRowCount();
	size_t count = GetElementCount();
	Values = new float[count];

	/*for (size_t i = 0; i < MaxValue; i++)
		Values[i] = c.GetValue(i);*/
	std::copy(c.Values, c.Values + count, Values);

#if USE_GPU
	cudaMalloc((void**)&GPUValues, sizeof(float) * MaxValue);
	CopyToGPU();
#endif // USE_GPU
}

Matrix::~Matrix()
{
	if (Values)
		delete[] Values;
#if USE_GPU
	cudaFree(GPUValues);
#endif
}

size_t Matrix::GetColumnCount() const
{
	return Columns;
}

size_t Matrix::GetRowCount() const
{
	return Rows;
}

float Matrix::GetValue(size_t row, size_t col) const
{
	size_t pos = RowColToPosition(row, col);
#if DEBUG
	if (pos < 0 || pos >= GetElementCount())
		return 0;
#endif // DEBUG
	return Values[pos];
}

float Matrix::GetValue(size_t pos) const
{
#if DEBUG
	if (pos < 0 || pos >= GetElementCount())
		throw MatrixIndexException();
#endif // DEBUG
	return Values[pos];
}

void Matrix::SetValue(size_t row, size_t col, float val)
{
	size_t pos = RowColToPosition(row, col);
#if DEBUG
	if (pos < 0 || pos >= GetElementCount())
		throw MatrixIndexException();
#endif // DEBUG
	Values[pos] = val;
}

void Matrix::SetValue(size_t pos, float val)
{
#if DEBUG
	if (pos < 0 || pos >= GetElementCount())
		throw MatrixIndexException();
#endif // DEBUG
	Values[pos] = val;
}

void Matrix::AdjustValue(size_t row, size_t col, float val)
{
	size_t pos = RowColToPosition(row, col);
#if DEBUG
	if (pos < 0 || pos >= GetElementCount())
		throw MatrixIndexException();
#endif // DEBUG == true
	Values[pos] += val;
}

void Matrix::AdjustValue(size_t pos, float val)
{
#if DEBUG
	if (pos < 0 || pos >= GetElementCount())
		throw MatrixIndexException();
#endif // DEBUG
	Values[pos] += val;
}

float Matrix::operator[](size_t id) const
{
#if DEBUG
	if (id < 0 || GetElementCount() <= id)
		throw MatrixIndexException();
#endif // DEBUG
	return Values[id];
}

Matrix& Matrix::operator=(const Matrix& other)
{
	if (this == &other)
		return *this;

	ReloadFromOther(other);

	return *this;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept
{
	if (this == &other)
		return *this;

	delete[] Values;
	Rows = std::exchange(other.Rows, 0);
	Columns = std::exchange(other.Columns, 0);
	Values = std::exchange(other.Values, nullptr);
	return *this;
}

Matrix& Matrix::operator+=(const Matrix& other)
{
	MatrixMath::AddIn(*this, other);
	return *this;
}

Matrix& Matrix::operator-=(const Matrix& other)
{
	MatrixMath::SubstractIn(*this, other);
	return *this;
}

Matrix& Matrix::operator*=(const Matrix& other)
{
	Matrix res = MatrixMath::Multiply(*this, other);
	ReloadFromOther(res);
	return *this;
}

Matrix Matrix::operator+(const Matrix& other) const
{
	Matrix res = MatrixMath::Add(*this, other);
	return res;
}

Matrix Matrix::operator-(const Matrix& other) const
{
	Matrix res = MatrixMath::Substract(*this, other);
	return res;
}

Matrix Matrix::operator*(const Matrix& other) const
{
	Matrix res = MatrixMath::Multiply(*this, other);
	return res;
}

bool Matrix::operator==(const Matrix& other) const
{
	return MatrixMath::IsEqual(*this, other);
}

bool Matrix::operator!=(const Matrix& other) const
{
	return !MatrixMath::IsEqual(*this, other);
}

Matrix& Matrix::operator*=(float other)
{
	MatrixMath::MultiplyIn(*this, other);
	return *this;
}

Matrix& Matrix::operator*(float other)
{
	Matrix res = MatrixMath::Multiply(*this, other);
	return res;
}

size_t Matrix::GetElementCount() const
{
	return Rows * Columns;
}

//std::ostream& Matrix::operator<<(std::ostream& os, const Matrix& m)
//{
//	// TODO: insert return statement here
//}

unsigned int Matrix::GetVectorSize()
{
	if (Columns == 1 && Rows > 1)
		return Rows;
	if (Rows == 1 && Columns > 1)
		return Columns;
#if DEBUG
	throw MatrixVectorException();
#endif // DEBUG

	return 0;
}

void Matrix::ReloadFromOther(const Matrix& m)
{
	delete[] Values;
	Columns = m.GetColumnCount();
	Rows = m.GetRowCount();
	size_t count = GetElementCount();
	Values = new float[count];

	std::copy(m.Values, m.Values + count, Values);

#if USE_GPU
	cudaFree(GPUValues);
	cudaMalloc((void**)&GPUValues, sizeof(float) * MaxValue);
	CopyToGPU();
#endif // USE_GPU

}

void Matrix::LoadFromJSON(const char* data, bool isFile)
{
	if (Values)
		delete[] Values;
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
	val = document["matrix"]["rows"];
	Rows = val.GetUint64();
	val = document["matrix"]["cols"];
	Columns = val.GetUint64();
	size_t MaxValue = GetElementCount();
	Values = new float[MaxValue];
	val = document["matrix"]["values"];
	size_t count = 0;
	for (rapidjson::Value::ConstValueIterator itr = val.Begin(); itr != val.End(); itr++)
	{
		Values[count] = itr->GetFloat();
		count++;
	}

#if USE_GPU
	cudaFree(GPUValues);
	cudaMalloc((void**)&GPUValues, sizeof(float) * MaxValue);
	CopyToGPU();
#endif
}

std::string Matrix::SaveToJSON(const char* fileName)
{
	rapidjson::Document doc;
	doc.SetObject();

	rapidjson::Value rows, cols, values;
	rapidjson::Value matrix(rapidjson::kObjectType);

	rows.SetUint64(Rows);
	cols.SetUint64(Columns);
	values.SetArray();
	for (size_t i = 0; i < Rows * Columns; i++)
		values.PushBack(Values[i], doc.GetAllocator());

	matrix.AddMember("rows", rows, doc.GetAllocator());
	matrix.AddMember("cols", cols, doc.GetAllocator());
	matrix.AddMember("values", values, doc.GetAllocator());

	doc.AddMember("matrix", matrix, doc.GetAllocator());

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

void Matrix::CopyToGPU()
{
#if USE_GPU
	cudaMemcpy((void*)GPUValues, (void*)Values, sizeof(float) * Rows * Columns, cudaMemcpyHostToDevice);
#endif
}

void Matrix::CopyFromGPU()
{
#if USE_GPU
	cudaMemcpy((void*)Values, (void*)GPUValues, sizeof(float) * Rows * Columns, cudaMemcpyDeviceToHost);
#endif
}

float* Matrix::GetGPUValues()
{
#if USE_GPU
	return GPUValues;
#else
	return nullptr;
#endif // 0
}

void Matrix::Reset(size_t rows, size_t columns)
{
	delete[] Values;
	Rows = rows;
	Columns = columns;
	Values = new float[Rows * Columns];
	std::fill(Values, Values + (rows * columns), 0);
}

inline size_t Matrix::RowColToPosition(size_t row, size_t col) const
{
	return row * Columns + col;
}
