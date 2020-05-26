#include "Matrix.h"
#include "Constants.h"
#include "MatrixException.hpp"

//TODO: Throw error at DEBUG errors

Matrix::Matrix()
{
	Columns = 1;
	Rows = 1;
	MaxValue = 1;
	Values = new float[1];
	Values[0] = 0;
}

Matrix::Matrix(size_t rows, size_t columns, float* elements)
{
	Columns = columns;
	Rows = rows;
	MaxValue = Rows * Columns;
	Values = new float[MaxValue];
	if (elements)
	{
		for (size_t i = 0; i < MaxValue; i++)
			Values[i] = elements[i];
	}
	else
	{
		for (size_t i = 0; i < MaxValue; i++)
			Values[i] = 0;
	}
}

Matrix::Matrix(const Matrix& c)
{
	Columns = c.GetColumnCount();
	Rows = c.GetRowCount();
	MaxValue = Rows * Columns;
	Values = new float[MaxValue];
	for (size_t i = 0; i < MaxValue; i++)
		Values[i] = c.GetValue(i);
}

Matrix::~Matrix()
{
	if (Values)
		delete[] Values;
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
	if (pos < 0 || pos >= MaxValue)
		return 0;
#endif // DEBUG
	return Values[pos];
}

float Matrix::GetValue(size_t pos) const
{
#if DEBUG
	if (pos < 0 || pos >= MaxValue)
		throw MatrixIndexException();
#endif // DEBUG
	return Values[pos];
}

void Matrix::SetValue(size_t row, size_t col, float val)
{
	size_t pos = RowColToPosition(row, col);
#if DEBUG
	if (pos < 0 || pos >= MaxValue)
		throw MatrixIndexException();
#endif // DEBUG
	Values[pos] = val;
}

void Matrix::SetValue(size_t pos, float val)
{
#if DEBUG
	if (pos < 0 || pos >= MaxValue)
		throw MatrixIndexException();
#endif // DEBUG
	Values[pos] = val;
}

void Matrix::AdjustValue(size_t row, size_t col, float val)
{
	size_t pos = RowColToPosition(row, col);
#if DEBUG
	if (pos < 0 || pos >= MaxValue)
		throw MatrixIndexException();
#endif // DEBUG == true
	Values[pos] += val;
}

void Matrix::AdjustValue(size_t pos, float val)
{
#if DEBUG
	if (pos < 0 || pos >= MaxValue)
		throw MatrixIndexException();
#endif // DEBUG
	Values[pos] += val;
}

float& Matrix::operator[](size_t id)
{
#if DEBUG
	if (id < 0 || MaxValue <= id)
		throw MatrixIndexException();
#endif // DEBUG
	return Values[id];
}

inline size_t Matrix::RowColToPosition(size_t row, size_t col) const
{
	return row * Columns + col;
}
