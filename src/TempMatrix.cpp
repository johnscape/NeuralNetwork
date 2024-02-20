#include "NeuralNetwork/TempMatrix.h"
#include "NeuralNetwork/MatrixException.hpp"
#include "NeuralNetwork/Constants.h"

TempMatrix::TempMatrix() : Matrix()
{}

TempMatrix::TempMatrix(size_t rows, size_t cols, unsigned int offset, float *values, float *gpu) : Matrix()
{
	Rows = rows;
	Columns = cols;
	Values = values + offset;
#if USE_GPU==USING_CUDA
    GPUValues = gpu + offset;
#else
    GPUValues = nullptr;
#endif
}

TempMatrix::~TempMatrix()
{
	Values = nullptr;
    GPUValues = nullptr;
}

Matrix TempMatrix::ToMatrix()
{
	return Matrix(Rows, Columns, Values);
}

void TempMatrix::Transpose()
{
	if (Rows == 1 || Columns == 1)
	{
		size_t tmp = Rows;
		Rows = Columns;
		Columns = tmp;
	}
	else
		throw StaticMatrixException();
}

Matrix TempMatrix::operator+(const Matrix &other) const
{
	return Matrix::operator+(other);
}

Matrix TempMatrix::operator-(const Matrix &other) const
{
	return Matrix::operator-(other);
}

Matrix TempMatrix::operator*(const Matrix &other) const
{
	return Matrix::operator*(other);
}

bool TempMatrix::operator==(const Matrix &other) const
{
	return Matrix::operator==(other);
}

bool TempMatrix::operator!=(const Matrix &other) const
{
	return Matrix::operator!=(other);
}

void TempMatrix::Pad(unsigned int top, unsigned int left, unsigned int bottom, unsigned int right, Matrix::PadType type,
					 float value, Matrix *result)
{
	if (result == nullptr)
		throw MatrixException();
	Matrix::Pad(top, left, bottom, right, type, value, result);
}

