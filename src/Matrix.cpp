#include "NeuralNetwork/Matrix.h"
#include "NeuralNetwork/Constants.h"
#include "NeuralNetwork/MatrixException.hpp"
#include "NeuralNetwork/Tensor.h"
#include <string>

#include <numeric>
#include <random>
#include <algorithm>
#include "nmmintrin.h"
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>
#include <fstream>

#if USE_GPU==USING_CUDA
#include <cuda_runtime.h>
#include "NeuralNetwork/CUDAMath.cuh"
#define CUDA_MALLOC(s) cudaMalloc((void**)&GPUValues, sizeof(float) * s)
#endif // USE_GPU

#define MATRIX_SIZE Rows * Columns

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

Matrix::Matrix() : GPUValues(nullptr), Values(nullptr), Columns(0), Rows(0)
{
}

Matrix::Matrix(size_t rows, size_t columns, float* elements) : GPUValues(nullptr)
{
	Columns = columns;
	Rows = rows;
	size_t count = GetElementCount();
	Values = new float[count];

	if (elements)
		std::copy(elements, elements + count, Values);
	else
		std::fill(Values, Values + count, 0);

#if USE_GPU==USING_CUDA
    MallocGPU();
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

#if USE_GPU==USING_CUDA
    MallocGPU();
	CopyToGPU();
#endif // USE_GPU
}

Matrix::Matrix(Matrix&& other) noexcept : GPUValues(other.GPUValues)
{
	Columns = other.Columns;
	Rows = other.Rows;
	Values = other.Values;
	GPUValues = other.GPUValues;

	other.Rows = 0;
	other.Columns = 0;
	other.Values = nullptr;
	other.GPUValues = nullptr;
}

Matrix::Matrix(const Tensor& from)
{
	if (from.GetShape().size() > 2)
		throw MatrixSizeException();
	Rows = from.GetShapeAt(0);
	Columns = from.GetShapeAt(1);
	Values = new float[Rows * Columns];
	for (unsigned int i = 0; i < Rows * Columns; ++i)
		Values[i] = from.GetValue(i); //TODO: Make it faster somehow, maybe make a const function, where we can use memcpy to copy values into a raw pointer e.g. Tensor::CopyValuesInto(float* values) {std::copy(v, v+size, values);}
    MallocGPU();
	CopyToGPU();
}

Matrix::~Matrix()
{

    delete[] Values;
    if (GPUValues)
	    FreeGPU();
}

TempMatrix Matrix::ToTempMatrix()
{
	return {Rows, Columns, Values};
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
    GPUValues = std::exchange(other.GPUValues, nullptr);
	return *this;
}

Matrix& Matrix::operator+=(const Matrix& other)
{
	if (!IsSameSize(other))
		throw MatrixException();
#if USE_GPU==USING_CUDA
	MatrixCUDAMath::AddIn(*this, other);
#else
	float floatRes[4];
	float currentValues[4];
	float otherValues[4];
	for (size_t i = 0; i < GetElementCount(); i += 4)
	{
		__m128 first, second;

		if (i + 4 < GetElementCount())
		{
			first = _mm_loadu_ps(Values + i);
			second = _mm_loadu_ps(other.Values + i);
		}
		else
		{
			for (unsigned char j = 0; j < 4; j++)
			{
				if (i + j < GetElementCount())
				{
					currentValues[j] = Values[i + j];
					otherValues[j] = other.Values[i + j];
				}
				else
				{
					currentValues[j] = 0;
					otherValues[j] = 0;
				}
			}
			first = _mm_load_ps(currentValues);
			second = _mm_load_ps(otherValues);
		}
		_mm_store_ps(floatRes, _mm_add_ps(first, second));
		size_t addressEnd = 4;
		if (i + addressEnd > GetElementCount())
			addressEnd = GetElementCount() - i;
		std::copy(floatRes, floatRes + addressEnd, Values + i);
	}
#endif
	return *this;
}

Matrix& Matrix::operator-=(const Matrix& other)
{
	if (!IsSameSize(other))
		throw MatrixException();
#if USE_GPU==USING_CUDA
	MatrixCUDAMath::SubtractIn(*this, other);
#else
	float floatRes[4];
	float currentValues[4];
	float otherValues[4];
	for (size_t i = 0; i < GetElementCount(); i += 4)
	{
		__m128 first, second;
		if (i + 4 < GetElementCount())
		{
			first = _mm_load_ps(Values + i);
			second = _mm_load_ps(other.Values + i);
		}
		else
		{
			for (unsigned char j = 0; j < 4; j++)
			{
				if (i + j < GetElementCount())
				{
					currentValues[j] = Values[i + j];
					otherValues[j] = other.Values[i + j];
				}
				else
				{
					currentValues[j] = 0;
					otherValues[j] = 0;
				}
			}
			first = _mm_load_ps(currentValues);
			second = _mm_load_ps(otherValues);
		}
		_mm_store_ps(floatRes, _mm_sub_ps(first, second));
		size_t addressEnd = 4;
		if (i + addressEnd > GetElementCount())
			addressEnd = GetElementCount() - i;
		std::copy(floatRes, floatRes + addressEnd, Values + i);
	}
#endif
	return *this;
}

Matrix& Matrix::operator*=(const Matrix& other)
{
	if (Columns != other.Rows)
		throw MatrixException();
    Matrix result(Rows, other.GetColumnCount());
#if USE_GPU==USING_CUDA
	MatrixCUDAMath::Multiplication(*this, other, result);
#else
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
#endif
    Rows = result.Rows;
    Columns = result.Columns;

    std::swap(Values, result.Values);
    std::swap(GPUValues, result.GPUValues);

	return *this;
}

Matrix Matrix::operator+(const Matrix& other) const
{
	if (!IsSameSize(other))
		throw MatrixException();
    Matrix result(Rows, Columns);
#if USE_GPU==USING_CUDA
	MatrixCUDAMath::Add(*this, other, result);
#else


	float floatRes[4];
	float currentValues[4];
	float otherValues[4];
	for (size_t i = 0; i < GetElementCount(); i += 4)
	{
		__m128 first, second;
		if (i + 4 < GetElementCount())
		{
			first = _mm_load_ps(Values + i);
			second = _mm_load_ps(other.Values + i);
		}
		else
		{
			for (unsigned char j = 0; j < 4; j++)
			{
				if (i + j < GetElementCount())
				{
					currentValues[j] = Values[i + j];
					otherValues[j] = other.Values[i + j];
				}
				else
				{
					currentValues[j] = 0;
					otherValues[j] = 0;
				}
			}
			first = _mm_load_ps(currentValues);
			second = _mm_load_ps(otherValues);
		}
		_mm_store_ps(floatRes, _mm_add_ps(first, second));
		size_t addressEnd = 4;
		if (i + addressEnd > GetElementCount())
			addressEnd = GetElementCount() - i;
		std::copy(floatRes, floatRes + addressEnd, result.Values + i);
	}
#endif
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const
{
	if (!IsSameSize(other))
		throw MatrixException();
    Matrix result(Rows, Columns);
#if USE_GPU==USING_CUDA
	MatrixCUDAMath::Subtract(*this, other, result);
#else


	float floatRes[4];
	float currentValues[4];
	float otherValues[4];
	for (size_t i = 0; i < GetElementCount(); i += 4)
	{
		__m128 first, second;
		if (i + 4 < GetElementCount())
		{
			first = _mm_load_ps(Values + i);
			second = _mm_load_ps(other.Values + i);
		}
		else
		{
			for (unsigned char j = 0; j < 4; j++)
			{
				if (i + j < GetElementCount())
				{
					currentValues[j] = Values[i + j];
					otherValues[j] = other.Values[i + j];
				}
				else
				{
					currentValues[j] = 0;
					otherValues[j] = 0;
				}
			}
			first = _mm_load_ps(currentValues);
			second = _mm_load_ps(otherValues);
		}
		_mm_store_ps(floatRes, _mm_sub_ps(first, second));
		size_t addressEnd = 4;
		if (i + addressEnd > GetElementCount())
			addressEnd = GetElementCount() - i;
		std::copy(floatRes, floatRes + addressEnd, result.Values + i);
	}
#endif
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const
{
	if (Columns != other.Rows)
		throw MatrixException();
    Matrix result(Rows, other.GetColumnCount());
#if USE_GPU==USING_CUDA
    MatrixCUDAMath::Multiplication(*this, other, result);
#else
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
#endif
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
#if USE_GPU==USING_CUDA
	MatrixCUDAMath::MultiplyConstant(*this, other);
#else
	for (size_t i = 0; i < GetElementCount(); i++)
		Values[i] *= other;
#endif
	return *this;
}

Matrix Matrix::operator*(float other)
{
	Matrix res(*this);
#if USE_GPU==USING_CUDA
	MatrixCUDAMath::MultiplyConstant(res, other);
#else
	for (size_t i = 0; i < GetElementCount(); i++)
		res.SetValue(i, res[i] * other);
#endif
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

inline bool Matrix::IsSquare() const
{
	return Rows == Columns;
}

Matrix Matrix::GetSubMatrix(size_t startRow, size_t startColumn, size_t rowNum, size_t colNum) const
{
	if (startColumn + colNum > Columns || startRow + rowNum > Rows)
		throw MatrixIndexException();
	if (rowNum == 0 || colNum == 0)
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

	return Matrix(1, Columns, Values + row * Columns);
}

TempMatrix Matrix::GetTempRowMatrix(size_t row) const
{
	if (row >= Rows)
		throw MatrixIndexException();
	return TempMatrix(1, Columns, Values + row * Columns);
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
#if USE_GPU==USING_CUDA
    MatrixCUDAMath::ElementwiseMultiply(c, b);
#else
	float floatRes[4];
	for (size_t i = 0; i < a.GetRowCount() * a.GetColumnCount(); i+=4)
	{
		__m128 first = _mm_load_ps(a.Values + i);
		__m128 second = _mm_load_ps(b.Values + i);
		_mm_store_ps(floatRes, _mm_mul_ps(first, second)); 
		size_t addressEnd = 4;
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
    eye.CopyToGPU();
	return eye;
}

Matrix Matrix::Concat(const Matrix& a, const Matrix& b, unsigned int dim)
{
	if (dim == 0) //add rows after each other
	{
		if (a.GetColumnCount() != b.GetColumnCount())
			throw MatrixSizeException();
		Matrix result(a.GetRowCount() + b.GetRowCount(), a.GetColumnCount());
		std::copy(a.Values, a.Values + a.GetElementCount(), result.Values);
		std::copy(b.Values, b.Values + b.GetElementCount(), result.Values + a.GetElementCount());
		return result;
	}
	else
	{
		if (a.GetRowCount() != b.GetRowCount())
			throw MatrixSizeException();
		Matrix result(a.GetRowCount(), a.GetColumnCount() + b.GetColumnCount());
		for (unsigned int row = 0; row < a.GetRowCount(); row++)
		{
			std::copy(
					a.Values + row * a.GetColumnCount(),
					a.Values + (row + 1) * a.GetColumnCount(),
					result.Values + row * result.GetColumnCount());
			std::copy(
					b.Values + row * b.GetColumnCount(),
					b.Values + (row + 1) * b.GetColumnCount(),
					result.Values + row * result.GetColumnCount() + a.GetColumnCount());
		}
		return result;
	}
}

Matrix Matrix::Concat(const Matrix& a, const Matrix& b, Matrix::ConcatType type)
{
	if (type == Matrix::ConcatType::BY_ROW)
		return Matrix::Concat(a, b, 0);
        return Matrix::Concat(a, b, 1);
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
		size_t addressEnd = 4;
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
		size_t addressEnd = 4;
		if (i + addressEnd > GetElementCount())
			addressEnd = GetElementCount() - i;
		for (unsigned int i = 0; i < addressEnd; i++)
			sum += floatRes[i];
	}

	return sum;
}

Matrix Matrix::Convolute(const Matrix& kernel, unsigned int stride, Matrix* target) const
{
	unsigned int rowDiff = Rows - kernel.GetRowCount();
	unsigned int colDiff = Columns - kernel.GetColumnCount();
	if (rowDiff % stride != 0 || colDiff % stride != 0)
		throw MatrixSizeException();

	unsigned int newRows = (rowDiff / stride) + 1;
	unsigned int newCols = (colDiff / stride) + 1;

	if (target && (target->GetRowCount() != newRows || target->GetColumnCount() != newCols))
			throw MatrixSizeException();


	Matrix result(newRows, newCols); //TODO: Do not create if thrown away
	for (unsigned int r = 0; r < newRows; ++r)
	{
		for (unsigned int c = 0; c < newCols; ++c)
		{
			for (unsigned int kr = 0; kr < kernel.GetRowCount(); ++kr)
			{
				for (int kc = 0; kc < kernel.GetColumnCount(); kc += 4)
				{
					float matValues[4] = {GetValue((r * stride) + kr, (c * stride) + kc),
										  0,
										  0,
										  0
					};
					float kerValues[4] = {kernel.GetValue(kr, kc),
										  0,
										  0,
										  0
					};
					for (unsigned int i = 1; i < 4; ++i)
					{
						if (kc + i >= kernel.GetColumnCount())
							break;
						matValues[i] = GetValue((r * stride) + kr, (c * stride) + kc + i);
						kerValues[i] = kernel.GetValue(kr, kc + i);
					}

					//Do a dot product on the vectors
					__m128 matReg = _mm_load_ps(matValues);
					__m128 kerReg = _mm_load_ps(kerValues);
					__m128 r1 = _mm_mul_ps(matReg, kerReg);
					__m128 shuf   = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 3, 0, 1));
					__m128 sums   = _mm_add_ps(r1, shuf);
					shuf          = _mm_movehl_ps(shuf, sums);
					sums          = _mm_add_ss(sums, shuf);
					float res =  _mm_cvtss_f32(sums);
					result.AdjustValue(r, c, res);
				}
			}
		}
	}

	if (target)
		target->ReloadFromOther(result);

	return result;
}

float Matrix::Sum() const
{
	return std::accumulate(Values, Values + GetElementCount(), 0);
}

float Matrix::Min() const
{
	return *std::min_element(Values, Values + GetElementCount());
}

float Matrix::Max() const
{
	return *std::max_element(Values, Values + GetElementCount());
}

void Matrix::Clamp(float min, float max)
{
	std::replace_if(Values, Values + GetElementCount(), [min](float val) {return val < min; }, min);
	std::replace_if(Values, Values + GetElementCount(), [max](float val) {return val > max; }, max);
}

void Matrix::RoundToInt()
{
	for (size_t i = 0; i < GetElementCount(); i++)
		Values[i] = roundf(Values[i]);
}

bool Matrix::IsOutOfBounds(size_t row, size_t col) const
{
	return (row >= Rows || col >= Columns);
}

void Matrix::Pad(unsigned int top, unsigned int left, unsigned int bottom, unsigned int right, PadType type,
				 float value, Matrix* result)
{
	Matrix padded(Rows + top + bottom, Columns + left + right);
	switch (type)
	{
	case Matrix::PadType::CONSTANT:
		padded.FillWith(value);
		break;
	default:
		break;
	}

	for (size_t i = 0; i < Rows; i++)
		std::copy(Values + i * Columns, Values + (i + 1) * Columns, padded.Values + left + (top + i) * padded.GetColumnCount());

	if (result == nullptr)
		ReloadFromOther(padded);
	else if (result->Columns == padded.Columns && result->Rows == padded.Rows)
		result->ReloadFromOther(padded);
	else
		throw MatrixSizeException();
}

void Matrix::ToSquare()
{
	if (Rows == Columns)
		return;
	if (Rows > Columns)
	{
		unsigned int diff = Rows - Columns;
		if ((diff & 0x01) == 0)
			Pad(0, diff / 2, 0, diff / 2);
		else
			Pad(0, (diff - 1) / 2 + 1, 0, (diff - 1) / 2);
	}
	else
	{
		unsigned int diff = Columns - Rows;
		if ((diff & 0x01) == 0)
			Pad(diff / 2, 0, diff / 2, 0);
		else
			Pad((diff - 1) / 2 + 1, 0, (diff - 1) / 2, 0);
	}
}

void Matrix::Rotate(unsigned int times)
{
	times = times % 4;
	if (times == 0)
		return;
	//If the matrix is square then we don't need anything special
	//But there might be a faster method
	if (IsSquare())
	{
		for (unsigned int t = 0; t < times; t++)
		{
			for (int i = 0; i < Rows / 2; i++)
			{
				for (int j = i; j < Rows - i - 1; j++)
				{
					float temp = GetValue(i, j);
					SetValue(i, j, GetValue(Rows - 1 - j, i));
					SetValue(Rows - 1 - j, i, GetValue(Rows - 1 - i, Rows - 1 - j));
					SetValue(Rows - 1 - i, Rows - 1 - j, GetValue(j, Rows - 1 - i));
					SetValue(j, Rows - 1 - i, temp);
				}
			}
		}
		return;
	}

	float* tmpArray = new float[Rows * Columns];
	if (times == 1 || times == 3)
	{
		for (unsigned int r = 0; r < Columns; ++r)
		{
			for (unsigned int c = 0; c < Rows; ++c)
			{
				if (times == 1)
					tmpArray[c + r * Rows] = Values[r + (Rows - 1 - c) * Columns];
				else
					tmpArray[c + r * Rows] = Values[c * Columns + (Columns - 1 - r)];
			}
		}
	}
	else
	{
		for (unsigned int r = 0; r < Rows; ++r)
		{
			std::copy(Values + r * Columns, Values + (r + 1) * Columns, tmpArray + (Rows - 1 - r) * Columns);
			std::reverse(tmpArray + (Rows - 1 - r) * Columns, tmpArray + (Rows - r) * Columns);
		}
	}

	delete[] Values;
	Values = tmpArray;
}

void Matrix::Normalize(float maxValue)
{
	float max = maxValue;
	if (max == 0)
	{
		float foundMax = Max();
		float foundMin = Min();
		if (foundMin < 0)
		{
			foundMin *= -1;
			max = foundMax > foundMin ? foundMax : foundMin;
		}
		else
			max = foundMax;
	}
	if (max < 0)
		max *= -1;

	for (size_t i = 0; i < GetElementCount(); i++)
		Values[i] /= max;
}

Matrix Matrix::Power(unsigned int p) const
{
	if (!IsSquare())
		throw MatrixException();
	if (p == 0)
		return Eye(Rows);
	Matrix res(Rows, Columns, Values);

	for (unsigned int v = 0; v < p - 1; v++)
        res *= (*this);


	return res;
}

void Matrix::PowerSelf(unsigned int p)
{
	if (!IsSquare())
		throw MatrixException();
	if (p == 0)
		ReloadFromOther(Eye(Rows));

	Matrix orig(*this);

	for (unsigned int v = 0; v < p - 1; v++)
		orig *= (*this);

	// ReloadFromOther(orig);
    Rows = orig.Rows;
    Columns = orig.Columns;
    std::swap(Values, orig.Values);
    std::swap(GPUValues, orig.GPUValues);

}

void Matrix::Transpose()
{
	if (Rows == 1 || Columns == 1)
	{
		size_t tmp = Rows;
		Rows = Columns;
		Columns = tmp;
		return;
	}
	Matrix trans(Columns, Rows);
	for (size_t r = 0; r < Rows; r++)
	{
		for (size_t c = 0; c < Columns; c++)
		{
			trans.SetValue(c, r, GetValue(r, c));
		}
	}
	
	ReloadFromOther(trans);
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
	size_t count = m.GetElementCount();
	if (count != GetElementCount())
	{
		if (Values)
			delete[] Values;
		Values = new float[count];
	}
	Columns = m.GetColumnCount();
	Rows = m.GetRowCount();
	std::copy(m.Values, m.Values + count, Values);

#if USE_GPU==USING_CUDA
	if (count != GetElementCount())
	{
		FreeGPU();
        GPUValues = nullptr;
        MallocGPU();
	}
	CopyToGPU();
#endif // USE_GPU

}

void Matrix::Reset(size_t rows, size_t columns)
{
	if (rows != Rows || columns != Columns)
	{
		delete[] Values;
		Rows = rows;
		Columns = columns;
		Values = new float[Rows * Columns];
#if USE_GPU==USING_CUDA
        FreeGPU();
        MallocGPU();
#endif
	}
	std::fill(Values, Values + (rows * columns), 0);
    CopyToGPU();
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
	#if USE_GPU==USING_CUDA
	MatrixCUDAMath::FillWith(*this, value);
	#endif
}

void Matrix::FillWithRandom(float min, float max)
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

	for (size_t i = 0; i < Rows * Columns; i++)
		Values[i] = dist(engine);
	#if USE_GPU==USING_CUDA
	CopyToGPU();
	#endif
}

void Matrix::LoadFromJSON(const char* data, bool isFile)
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
	value = document["matrix"];
	LoadFromJSON(value);
}

void Matrix::LoadFromJSON(rapidjson::Value& jsonValue)
{
	if (Values)
		delete[] Values;
	if (jsonValue.HasMember("matrix"))
		jsonValue = jsonValue["matrix"];
	Rows = jsonValue["rows"].GetUint64();
	Columns = jsonValue["cols"].GetUint64();
	rapidjson::Value arr;
	arr = jsonValue["values"];
	Values = new float[Rows * Columns];
	unsigned int counter = 0;
	for (rapidjson::Value::ConstValueIterator itr = arr.Begin(); itr != arr.End(); itr++)
	{
		Values[counter] = itr->GetFloat();
		counter++;
	}
}

std::string Matrix::SaveToJSON(const char* fileName) const
{
	rapidjson::Document document;
	document.SetObject();
	rapidjson::Value matrix = SaveToJSONObject(document);

	document.AddMember("matrix", matrix, document.GetAllocator());

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

rapidjson::Value Matrix::SaveToJSONObject(rapidjson::Document& document) const
{
	//create objects
	rapidjson::Value rows, cols, values;
	rapidjson::Value matrix(rapidjson::kObjectType);

	//fill objects with values
	rows.SetUint64(Rows);
	cols.SetUint64(Columns);
	values.SetArray();
	for (unsigned int i = 0; i < GetElementCount(); ++i)
		values.PushBack(Values[i], document.GetAllocator());

	//add inheritance
	matrix.AddMember("rows", rows, document.GetAllocator());
	matrix.AddMember("cols", cols, document.GetAllocator());
	matrix.AddMember("values", values, document.GetAllocator());

	return matrix;
}

void Matrix::CopyToGPU()
{
#if USE_GPU==USING_CUDA
    cudaError_t err = cudaMemcpy((void*)GPUValues, (void*)Values, sizeof(float) * MATRIX_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        std::cerr << "CUDA error: " << cudaGetErrorName(err) << " - " << cudaGetErrorString(err) << std::endl;
#endif
}

void Matrix::CopyFromGPU()
{
#if USE_GPU==USING_CUDA
    cudaError_t err = cudaMemcpy((void*)Values, (void*)GPUValues, sizeof(float) * MATRIX_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        std::cerr << "CUDA error: " << cudaGetErrorName(err) << " - " << cudaGetErrorString(err) << std::endl;
#endif
}

float* Matrix::GetGPUValues()
{
#if USE_GPU==USING_CUDA
	return GPUValues;
#else
	return nullptr;
#endif // 0
}

float* Matrix::GetConstGPUValues() const
{
#if USE_GPU==USING_CUDA
	return GPUValues;
#else
	return nullptr;
#endif // 0
}

inline size_t Matrix::RowColToPosition(size_t row, size_t col) const
{
	return row * Columns + col;
}

void Matrix::MallocGPU()
{
#if USE_GPU==USING_CUDA
    if (GPUValues)
        std::cerr << "GPU address is not null, possible memory leak!" << std::endl;
    cudaError_t err = cudaMalloc(&GPUValues, sizeof(float) * MATRIX_SIZE);
    if (err != cudaSuccess)
        std::cerr << "CUDA error: " << cudaGetErrorName(err) << " - " << cudaGetErrorString(err) << std::endl;
#endif
}

void Matrix::FreeGPU()
{
#if USE_GPU==USING_CUDA
    cudaError_t err = cudaFree(GPUValues);
    if (err != cudaSuccess)
        std::cerr << "CUDA error: " << cudaGetErrorName(err) << " - " << cudaGetErrorString(err) << std::endl;
    GPUValues = nullptr;
#endif
}
