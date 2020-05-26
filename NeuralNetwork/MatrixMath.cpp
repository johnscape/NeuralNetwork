#include "MatrixMath.h"
#include <iostream>

bool MatrixMath::SizeCheck(const Matrix* a, const Matrix* b)
{
	return (a->GetColumnCount() == b->GetColumnCount()) && (a->GetRowCount() == b->GetRowCount());
}

bool MatrixMath::IsVector(Matrix* matrix)
{
	return (matrix->GetColumnCount() == 1 || matrix->GetRowCount() == 1);
}

bool MatrixMath::IsEqual(Matrix* a, Matrix* b)
{
	if (!SizeCheck(a, b))
		throw MatrixSizeException();
	for (size_t r = 0; r < a->GetRowCount(); r++)
	{
		for (size_t c = 0; c < a->GetColumnCount(); c++)
		{
			if (a->GetValue(r, c) != b->GetValue(r, c))
				return false;
		}
	}

	return true;
}

void MatrixMath::FillWith(Matrix* m, float value)
{
	for (unsigned int i = 0; i < m->GetColumnCount() * m->GetRowCount(); i++)
		m->SetValue(i, value);
}

void MatrixMath::FillWithRandom(Matrix* m)
{

}

void MatrixMath::Copy(Matrix* from, Matrix* to)
{
	for (size_t i = 0; i < to->GetColumnCount() * to->GetRowCount(); i++)
		to->SetValue(i, (*from)[i]);
}

void MatrixMath::AddIn(Matrix* a, Matrix* b)
{
	if (!SizeCheck(a, b))
		throw MatrixSizeException();
	for (unsigned int i = 0; i < a->GetColumnCount() * a->GetRowCount(); i++)
		a->AdjustValue(i, (*b)[i]);
}

void Add(Matrix* matrix, float value)
{
	for (size_t i = 0; i < matrix->GetColumnCount() * matrix->GetRowCount(); i++)
		matrix->AdjustValue(i, value);
}

Matrix* MatrixMath::Add(const Matrix* a, const Matrix* b)
{
	if (!SizeCheck(a, b))
		throw MatrixException(); //the two matrices doesn't have the same column/row count!
	Matrix* c = new Matrix(a->GetRowCount(), a->GetColumnCount());
	for (size_t i = 0; i < a->GetRowCount() * a->GetColumnCount(); i++)
		c->SetValue(i, a->GetValue(i) + b->GetValue(i));
	return c;
}

Matrix* MatrixMath::Substract(Matrix* a, Matrix* b)
{
	if (!SizeCheck(a, b))
		throw MatrixException(); //the two matrices doesn't have the same column/row count!
	Matrix* c = new Matrix(a->GetRowCount(), a->GetColumnCount());
	for (size_t i = 0; i < a->GetRowCount() * a->GetColumnCount(); i++)
		c->SetValue(i, (*a)[i] - (*b)[i]);
	return c;
}

void MatrixMath::MultiplyIn(Matrix* a, float b)
{
	for (size_t i = 0; i < a->GetColumnCount() * a->GetRowCount(); i++)
		a->SetValue(i, (*a)[i] * b);
}

Matrix* MatrixMath::Multiply(Matrix* a, float b)
{
	Matrix* c = new Matrix(a->GetRowCount(), a->GetColumnCount());
	for (size_t i = 0; i < a->GetColumnCount() * a->GetRowCount(); i++)
		c->SetValue(i, (*a)[i] * b);
	return c;
}

Matrix* MatrixMath::Multiply(Matrix* a, Matrix* b, Matrix* c)
{
	if (!c)
		Matrix* c = new Matrix(a->GetRowCount(), a->GetColumnCount());
	CacheVector col, row, temp;
	size_t br;
	for (size_t bc = 0; bc < b->GetColumnCount(); bc++)
	{
		br = 0;
		while (br < b->GetRowCount())
		{
			for (unsigned char i = 0; i < 4; i++)
				col.fl[i] = b->GetValue(br + i, bc);
			for (size_t ar = 0; ar < a->GetRowCount(); ar++)
			{
				for (size_t i = 0; i < 4; i++)
					row.fl[i] = a->GetValue(ar, br + i);
				temp.vec = _mm_dp_ps(col.vec, row.vec, 0xff);
				c->AdjustValue(ar, bc, temp.fl[0]);
			}

			br += 4;
		}

	}
	return c;
}

Matrix* MatrixMath::SlowMultiply(Matrix* a, Matrix* b)
{
	Matrix* c = new Matrix(a->GetRowCount(), b->GetColumnCount());
	MatrixMath::FillWith(c, 0);
	for (unsigned int i = 0; i < a->GetRowCount(); i++)
		for (unsigned int j = 0; j < b->GetColumnCount(); j++)
			for (unsigned int k = 0; k < a->GetColumnCount(); k++)
				c->AdjustValue(i, j, a->GetValue(i, k) * b->GetValue(k, j));
	return c;
}

void MatrixMath::Transpose(Matrix* m)
{
	if (m->GetColumnCount() == m->GetRowCount())
	{
		size_t count = m->GetColumnCount();
		for (size_t r = 0; r < count; r++)
		{
			for (size_t c = 0; c < count; c++)
			{
				if (r >= c)
					continue;
				float temp = m->GetValue(r, c);
				m->SetValue(r, c, m->GetValue(c, r));
				m->SetValue(c, r, temp);
			}
		}
	}
	else
	{
		Matrix trans(m->GetColumnCount(), m->GetRowCount());
		for (unsigned int i = 0; i < m->GetRowCount(); i++)
			for (unsigned int j = 0; j < m->GetColumnCount(); j++)
				trans.SetValue(j, i, m->GetValue(i, j));
		Copy(&trans, m);
	}
}

Matrix* MatrixMath::GetRowMatrix(Matrix* m, size_t row)
{
	Matrix* rowm = new Matrix(1, m->GetColumnCount());
	for (size_t i = 0; i < m->GetColumnCount(); i++)
		rowm->SetValue(i, m->GetValue(row, i));
	return rowm;
}

Matrix* MatrixMath::GetColumnMatrix(Matrix* m, size_t column)
{
	Matrix* colm = new Matrix(m->GetRowCount(), 1);
	for (size_t i = 0; i < m->GetRowCount(); i++)
		colm->SetValue(i, m->GetValue(i, column));
	return colm;
}

Matrix* MatrixMath::Hadamard(Matrix* a, Matrix* b)
{
	if (!SizeCheck(a, b))
		throw MatrixException(); //the two matrices doesn't have the same column/row count!
	Matrix* cmat = new Matrix(a->GetRowCount(), a->GetColumnCount());
	for (size_t r = 0; r < a->GetRowCount(); r++)
		for (size_t c = 0; c < a->GetColumnCount(); c++)
			cmat->SetValue(r, c, a->GetValue(r, c) * b->GetValue(r, c));
	return cmat;
}

Matrix* MatrixMath::OuterProduct(Matrix* a, Matrix* b)
{
	if ((!IsVector(a) && !IsVector(b)) || (a->GetRowCount() == b->GetRowCount()))
		throw MatrixException(); //the two matrices must be vectors
	Matrix* cmat;
	if (a->GetRowCount() == 1)
	{
		cmat = new Matrix(b->GetRowCount(), a->GetColumnCount());
		for (size_t r = 0; r < b->GetRowCount(); r++)
			for (size_t c = 0; c < a->GetColumnCount(); c++)
				cmat->SetValue(r, c, (*b)[r] * (*a)[c]);
	}
	else
	{
		cmat = new Matrix(a->GetRowCount(), b->GetColumnCount());
		for (size_t r = 0; r < a->GetRowCount(); r++)
			for (size_t c = 0; c < b->GetColumnCount(); c++)
				cmat->SetValue(r, c, (*a)[r] * (*b)[c]);
	}

	return cmat;
}

Matrix* MatrixMath::CreateSubMatrix(Matrix* m, size_t startRow, size_t startCol, size_t rowCount, size_t colCount)
{
	if (startRow + rowCount >= m->GetRowCount() || startCol + colCount >= m->GetColumnCount())
		throw MatrixException();
	Matrix* mat = new Matrix(rowCount, colCount);
	for (size_t row = 0; row < rowCount; row++)
	{
		for (size_t col = 0; col < colCount; col++)
		{
			mat->SetValue(row, col, m->GetValue(startRow + row, startCol + col));
		}
	}

	return mat;
}

float MatrixMath::DotProduct(Matrix* a, Matrix* b)
{
	if ((!IsVector(a) && !IsVector(b)))
		throw MatrixException(); //the two matrices must be vectors
	if (b->GetColumnCount() * b->GetRowCount() != a->GetColumnCount() * a->GetRowCount())
		throw MatrixException();

	float value = 0;
	for (size_t i = 0; i < a->GetColumnCount() * a->GetRowCount(); i++)
		value += (*a)[i] * (*b)[i];
	return value;
}

float MatrixMath::Sum(Matrix* m)
{
	float value = 0;
	for (size_t i = 0; i < m->GetColumnCount() * m->GetRowCount(); i++)
		value += (*m)[i];
	return value;
}

Matrix* MatrixMath::Eye(unsigned int size)
{
	Matrix* m = new Matrix(size, size);
	for (size_t i = 0; i < size; i++)
		m->SetValue(i, i, 1);
	return m;
}

void MatrixMath::PrintMatrix(Matrix* m)
{
	for (unsigned int r = 0; r < m->GetRowCount(); r++)
	{
		for (unsigned int c = 0; c < m->GetColumnCount(); c++)
		{
			std::cout << m->GetValue(r, c) << " ";
		}
		std::cout << std::endl;
	}
}
