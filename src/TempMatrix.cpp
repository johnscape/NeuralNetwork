//
// Created by attila on 2022. 03. 10..
//

#include "NeuralNetwork/TempMatrix.h"

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

