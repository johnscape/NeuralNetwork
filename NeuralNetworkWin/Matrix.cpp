#include "Matrix.h"
#include "Constants.h"
#include "MatrixException.hpp"
#include <fstream>
#include <string>

#include <numeric>
#include <functional>
#include <random>
#include "nmmintrin.h"
#include "immintrin.h"

#if USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif // USE_GPU

#define ROUND_UP(x, s) (((x)+((s)-1)) & -(s))

std::ostream& operator<<(std::ostream& os, const Matrix& m)
{
	for (size_t row = 0; row < m.GetRowCount(); row++)
	{
		for (size_t col = 0; col < m.GetColumnCount(); col++)
		{
			os << m.GetValue(row, col) << '\t';
		}

		os << std::endl;
	}

	return os;
}

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

Matrix::Matrix(const Matrix& other) : GPUValues(nullptr)
{
	Columns = other.GetColumnCount();
	Rows = other.GetRowCount();
	size_t count = GetElementCount();
	Values = new float[count];

	std::copy(other.Values, other.Values + count, Values);

#if USE_GPU
	cudaMalloc((void**)&GPUValues, sizeof(float) * MaxValue);
	CopyToGPU();
#endif // USE_GPU
}

Matrix::Matrix(Matrix&& other) noexcept : GPUValues(other.GPUValues)
{
	Columns = other.Columns;
	Rows = other.Rows;
	Values = other.Values;

	other.Rows = 0;
	other.Columns = 0;
	other.Values = nullptr;
	other.GPUValues = nullptr;
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

bool Matrix::IsVector() const
{
	return Rows == 1 || Columns == 1;
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
	if (!IsSameSize(other))
		throw MatrixException();

	float floatRes[4];
	for (size_t i = 0; i < GetElementCount(); i += 4)
	{
		__m128 first = _mm_load_ps(Values + i);
		__m128 second = _mm_load_ps(other.Values + i);
		_mm_store_ps(floatRes, _mm_add_ps(first, second));
		unsigned int addressEnd = 4;
		if (i + addressEnd > GetElementCount())
			addressEnd = GetElementCount() - i;
		std::copy(floatRes, floatRes + addressEnd, Values + i);
	}

	return *this;
}

Matrix& Matrix::operator-=(const Matrix& other)
{
	if (!IsSameSize(other))
		throw MatrixException();

	float floatRes[4];
	for (size_t i = 0; i < GetElementCount(); i += 4)
	{
		__m128 first = _mm_load_ps(Values + i);
		__m128 second = _mm_load_ps(other.Values + i);
		_mm_store_ps(floatRes, _mm_sub_ps(first, second));
		unsigned int addressEnd = 4;
		if (i + addressEnd > GetElementCount())
			addressEnd = GetElementCount() - i;
		std::copy(floatRes, floatRes + addressEnd, Values + i);
	}
	return *this;
}

Matrix& Matrix::operator*=(const Matrix& other)
{
	if (Columns != other.Rows)
		throw MatrixException();

	Matrix result(Rows, other.GetColumnCount());
	//CacheVector col, row;
	float col[4];
	float row[4];
	size_t br;
	for (size_t bc = 0; bc < other.GetColumnCount(); bc++)
	{
		br = 0;
		while (br < other.GetRowCount())
		{
			for (unsigned char i = 0; i < 4; i++)
				col[i] = other.GetValue(br + i, bc);
			__m128 colVec = _mm_load_ps(col);
			for (size_t ar = 0; ar < GetRowCount(); ar++)
			{
				for (size_t i = 0; i < 4; i++)
					row[i] = GetValue(ar, br + i);
				__m128 rowVec = _mm_load_ps(row);
				__m128 mul = _mm_mul_ps(colVec, rowVec);
				__m128 shuf = _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(2, 3, 0, 1));
				__m128 sums = _mm_add_ps(mul, shuf);
				shuf = _mm_movehl_ps(shuf, sums);
				sums = _mm_add_ss(sums, shuf);
				float res = _mm_cvtss_f32(sums);

				result.AdjustValue(ar, bc, res);
			}
			br += 4;
		}
	}

	ReloadFromOther(result);

	return *this;
}

Matrix Matrix::operator+(const Matrix& other) const
{
	if (!IsSameSize(other))
		throw MatrixException();

	Matrix res(Rows, Columns);

	float floatRes[4];
	for (size_t i = 0; i < GetElementCount(); i += 4)
	{
		__m128 first = _mm_load_ps(Values + i);
		__m128 second = _mm_load_ps(other.Values + i);
		_mm_store_ps(floatRes, _mm_add_ps(first, second));
		unsigned int addressEnd = 4;
		if (i + addressEnd > GetElementCount())
			addressEnd = GetElementCount() - i;
		std::copy(floatRes, floatRes + addressEnd, res.Values + i);
	}

	return res;
}

Matrix Matrix::operator-(const Matrix& other) const
{
	if (!IsSameSize(other))
		throw MatrixException();

	Matrix res(Rows, Columns);

	float floatRes[4];
	for (size_t i = 0; i < GetElementCount(); i += 4)
	{
		__m128 first = _mm_load_ps(Values + i);
		__m128 second = _mm_load_ps(other.Values + i);
		_mm_store_ps(floatRes, _mm_sub_ps(first, second));
		unsigned int addressEnd = 4;
		if (i + addressEnd > GetElementCount())
			addressEnd = GetElementCount() - i;
		std::copy(floatRes, floatRes + addressEnd, res.Values + i);
	}

	return res;
}

Matrix Matrix::operator*(const Matrix& other) const
{
	if (Columns != other.Rows)
		throw MatrixException();

	Matrix result(Rows, other.GetColumnCount());
	//CacheVector col, row;
	float col[4];
	float row[4];
	size_t br;
	for (size_t bc = 0; bc < other.GetColumnCount(); bc++)
	{
		br = 0;
		while (br < other.GetRowCount())
		{
			for (unsigned char i = 0; i < 4; i++)
				col[i] = other.GetValue(br + i, bc);
			__m128 colVec = _mm_load_ps(col);
			for (size_t ar = 0; ar < GetRowCount(); ar++)
			{
				for (size_t i = 0; i < 4; i++)
					row[i] = GetValue(ar, br + i);
				__m128 rowVec = _mm_load_ps(row);
				__m128 mul = _mm_mul_ps(colVec, rowVec);
				__m128 shuf = _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(2, 3, 0, 1));
				__m128 sums = _mm_add_ps(mul, shuf);
				shuf = _mm_movehl_ps(shuf, sums);
				sums = _mm_add_ss(sums, shuf);
				float res = _mm_cvtss_f32(sums);

				result.AdjustValue(ar, bc, res);
			}
			br += 4;
		}
	}

	return result;
}

bool Matrix::operator==(const Matrix& other) const
{
	if (!IsSameSize(other))
		return false;
	for (unsigned int i = 0; i < GetElementCount(); i++)
		if (Values[i] != other[i])
			return false;

	return true;
}

bool Matrix::operator!=(const Matrix& other) const
{
	if (!IsSameSize(other))
		return true;
	for (unsigned int i = 0; i < GetElementCount(); i++)
		if (Values[i] != other[i])
			return true;

	return false;
}

Matrix& Matrix::operator*=(float other)
{
	for (size_t i = 0; i < GetElementCount(); i++)
		Values[i] *= other;
	return *this;
}

Matrix Matrix::operator*(float other)
{
	Matrix res(*this);

	for (size_t i = 0; i < GetElementCount(); i++)
		res.SetValue(i, res[i] * other);

	return res;
}

size_t Matrix::GetElementCount() const
{
	return Rows * Columns;
}

inline bool Matrix::IsSameSize(const Matrix& other) const
{
	return other.Columns == Columns && other.Rows == Rows;
}

Matrix Matrix::GetSubMatrix(size_t startRow, size_t startColumn, size_t rowNum, size_t colNum) const
{
	if (startColumn + colNum >= Columns || startRow + rowNum >= Rows)
		throw MatrixIndexException();

	Matrix sub(rowNum, colNum);

	for (size_t r = 0; r < rowNum; r++)
	{
		size_t startValue = (startRow + r) * Columns + startColumn;
		size_t endValue = (startRow + r) * Columns + startColumn + colNum;
		std::copy(Values + startValue, Values + endValue, sub.Values + r * colNum);
	}

	return sub;
}

Matrix Matrix::GetRowMatrix(size_t row) const
{
	if (row >= Rows)
		throw MatrixIndexException();


	Matrix rowmat(1, Columns);
	std::copy(Values + row * Columns, Values + (row + 1) * Columns, rowmat.Values);
	return rowmat;
}

Matrix Matrix::GetColumnMatrix(size_t col) const
{
	if (col >= Columns)
		throw MatrixIndexException();

	Matrix colmat(Rows, 1);
	for (size_t i = 0; i < Rows; i++)
		colmat.SetValue(i, GetValue(i, col));
	return colmat;
}

Matrix Matrix::ElementwiseMultiply(const Matrix& a, const Matrix& b)
{
	if (!a.IsSameSize(b))
		throw MatrixException();
	Matrix c(a);
#if USE_GPU

#else
	float floatRes[4];
	for (size_t i = 0; i < a.GetRowCount() * a.GetColumnCount(); i+=4)
	{
		__m128 first = _mm_load_ps(a.Values + i);
		__m128 second = _mm_load_ps(b.Values + i);
		_mm_store_ps(floatRes, _mm_mul_ps(first, second)); 
		unsigned int addressEnd = 4;
		if (i + addressEnd > a.GetRowCount() * a.GetColumnCount())
			addressEnd = (a.GetRowCount() * a.GetColumnCount()) - i;
		std::copy(floatRes, floatRes + addressEnd, c.Values + i);
	}
#endif
	return c;
}

Matrix Matrix::Eye(unsigned int i)
{
	Matrix eye(i, i);
	for (unsigned int j = 0; j < i; j++)
		eye.SetValue(j, j, 1);
	return eye;
}

Matrix Matrix::Concat(const Matrix& a, const Matrix& b, unsigned int dim)
{
	return Matrix();
}

void Matrix::ElementwiseMultiply(const Matrix& other)
{
	if (!IsSameSize(other))
		throw MatrixException();
	float floatRes[4];
	for (size_t i = 0; i < GetElementCount(); i += 4)
	{
		__m128 first = _mm_load_ps(Values + i);
		__m128 second = _mm_load_ps(other.Values + i);
		_mm_store_ps(floatRes, _mm_mul_ps(first, second));
		unsigned int addressEnd = 4;
		if (i + addressEnd > GetElementCount())
			addressEnd = GetElementCount() - i;
		std::copy(floatRes, floatRes + addressEnd, Values + i);
	}
}

Matrix Matrix::OuterProduct(const Matrix& vector) const
{
	if (!IsVector() || !vector.IsVector())
		throw MatrixException();
	Matrix outer(GetVectorSize(), vector.GetVectorSize());

	for (unsigned int r = 0; r < outer.GetRowCount(); r++)
		for (unsigned int c = 0; c < outer.GetColumnCount(); c++)
			outer.SetValue(r, c, GetValue(r) * vector.GetValue(c));

	return outer;
}

float Matrix::DotProcudt(const Matrix& vector) const
{
	if (!IsVector() || !vector.IsVector())
		throw MatrixException();

	float sum = 0;
	float floatRes[4];
	for (size_t i = 0; i < GetElementCount(); i += 4)
	{
		__m128 first = _mm_load_ps(Values + i);
		__m128 second = _mm_load_ps(vector.Values + i);
		_mm_store_ps(floatRes, _mm_mul_ps(first, second));
		unsigned int addressEnd = 4;
		if (i + addressEnd > GetElementCount())
			addressEnd = GetElementCount() - i;
		for (unsigned int i = 0; i < addressEnd; i++)
			sum += floatRes[i];
	}

	return sum;
}

float Matrix::Sum() const
{
	return std::accumulate(Values, Values + GetElementCount(), 0);
}

Matrix Matrix::Power(unsigned int p) const
{
	return Matrix();
}

void Matrix::PowerSelf(unsigned int p)
{
}

void Matrix::Transpose()
{
	TransposeArray(Values, Columns, Rows);
	size_t tmp = Rows;
	Rows = Columns;
	Columns = tmp;
}

size_t Matrix::GetVectorSize() const
{
	if (Columns == 1)
		return Rows;
	if (Rows == 1)
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

void Matrix::Reset(size_t rows, size_t columns)
{
	delete[] Values;
	Rows = rows;
	Columns = columns;
	Values = new float[Rows * Columns];
	std::fill(Values, Values + (rows * columns), 0);
}

void Matrix::Copy(const Matrix& from)
{
	if (Rows != from.Rows || Columns != from.Columns)
		throw MatrixException();
	std::copy(from.Values, from.Values + Rows * Columns, Values);
}

void Matrix::FillWith(float value)
{
	std::fill(Values, Values + Rows * Columns, value);
}

void Matrix::FillWithRandom(float min, float max)
{
	static std::random_device device;
	static std::mt19937 engine;
	engine.seed(device());
	std::uniform_real_distribution<> dist(min, max);

	for (size_t i = 0; i < Rows * Columns; i++)
		Values[i] = dist(engine);
}

void Matrix::LoadFromJSON(const char* data, bool isFile)
{
	/*if (Values)
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
#endif*/
}

std::string Matrix::SaveToJSON(const char* fileName)
{
	/*rapidjson::Document doc;
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

	return std::string(buffer.GetString());*/

	return "";
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

inline size_t Matrix::RowColToPosition(size_t row, size_t col) const
{
	return row * Columns + col;
}

void Matrix::TransposeArray(float* arr, unsigned int w, unsigned int h) const
{
	/*unsigned int lda = ROUND_UP(w, 16);
	unsigned int ldb = ROUND_UP(h, 16);

	float* A = new float[lda * ldb];
	float* B = new float[lda * ldb];

	std::copy(arr, arr + w * h, A);
	std::fill(B, B + lda * ldb, 0);

#pragma omp parallel for
	for (unsigned int i = 0; i < w; i += 4)
	{
		for (unsigned int j = 0; j < h; j += 4)
		{
			int max_i2 = i + 4 < h ? i + 4 : h;
			int max_j2 = j + 4 < w ? j + 4 : w;

			for (int i2 = i; i2 < max_i2; i2 += 4)
				for (int j2 = j; j2 < max_j2; j2 += 4)
					TransposeBlock(&A[i2 * lda + j2], &B[j2 * ldb + i2], lda, ldb);
		}
	}

	std::copy(B, B + GetElementCount(), arr);

	delete[] A;
	delete[] B;*/
}

void Matrix::TransposeBlock(float* A, float* B, unsigned int lda, unsigned int ldb) const
{
	__m128 row1 = _mm_load_ps(&A[0 * lda]);
	__m128 row2 = _mm_load_ps(&A[1 * lda]);
	__m128 row3 = _mm_load_ps(&A[2 * lda]);
	__m128 row4 = _mm_load_ps(&A[3 * lda]);
	_MM_TRANSPOSE4_PS(row1, row2, row3, row4);
	_mm_store_ps(&B[0 * ldb], row1);
	_mm_store_ps(&B[1 * ldb], row2);
	_mm_store_ps(&B[2 * ldb], row3);
	_mm_store_ps(&B[3 * ldb], row4);
}
