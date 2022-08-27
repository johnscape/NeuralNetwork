#include "NeuralNetwork/TempMatrix.h"
#include "NeuralNetwork/MatrixException.hpp"

TempMatrix::TempMatrix() : Matrix()
{}

TempMatrix::TempMatrix(size_t rows, size_t cols, float *values) : Matrix()
{
	Rows = rows;
	Columns = cols;
	Values = values;
}

TempMatrix::~TempMatrix()
{
	Values = nullptr;
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

