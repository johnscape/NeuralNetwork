#include "Matrix.h"
#include "Constants.h"
#include "MatrixException.hpp"
#include <fstream>
#include <string>

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

unsigned int Matrix::GetVectorSize()
{
	if (Columns == 1 && Rows > 1)
		return Rows;
	if (Rows == 1 && Columns > 1)
		return Columns;
#ifdef DEBUG
	throw MatrixVectorException();
#endif // DEBUG

	return 0;
}

void Matrix::ReloadFromOther(Matrix* m)
{
	delete[] Values;
	Columns = m->GetColumnCount();
	Rows = m->GetRowCount();
	MaxValue = Rows * Columns;
	Values = new float[MaxValue];
	for (size_t i = 0; i < MaxValue; i++)
		Values[i] = m->GetValue(i);
}

void Matrix::SaveToFile(const char* fileName)
{
	std::ofstream writer(fileName);
	writer << Rows << " " << Columns;
	for (size_t i = 0; i < Rows * Columns; i++)
		writer << " " << Values[i];
	writer.close();
}

void Matrix::LoadFromFile(const char* fileName)
{
	if (Values)
		delete[] Values;
	std::ifstream reader(fileName);
	std::string txt;
	Rows = 0;
	Columns = 0;
	size_t count = 0;
	while (std::getline(reader, txt, ' '))
	{
		if (!Rows)
			Rows = std::stoi(txt);
		else if (!Columns)
		{
			Columns = std::stoi(txt);
			Values = new float[Rows * Columns];
		}
		else
		{
			Values[count] = std::stof(txt);
			count++;
		}
	}
	reader.close();
}

inline size_t Matrix::RowColToPosition(size_t row, size_t col) const
{
	return row * Columns + col;
}
