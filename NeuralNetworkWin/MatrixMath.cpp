#include "MatrixMath.h"
#include <iostream>
#include "Constants.h"
#include "MatrixException.hpp"
#include "MatrixGPUMath.cuh"

bool MatrixMath::SizeCheck(const Matrix* a, const Matrix* b)
{
	return (a->GetColumnCount() == b->GetColumnCount()) && (a->GetRowCount() == b->GetRowCount());
}

bool MatrixMath::IsVector(const Matrix* matrix)
{
	return (matrix->GetColumnCount() == 1 || matrix->GetRowCount() == 1);
}

bool MatrixMath::IsEqual(const Matrix* a, const Matrix* b)
{
	if (!SizeCheck(a, b))
		return false;
	for (size_t i = 0; i < a->GetColumnCount() * a->GetRowCount(); i++)
	{
		if (a->GetValue(i) != b->GetValue(i))
			return false;
	}
	return true;
}

void MatrixMath::FillWith(Matrix* m, float value)
{
	for (unsigned int i = 0; i < m->GetColumnCount() * m->GetRowCount(); i++)
		m->SetValue(i, value);
}

void MatrixMath::FillWithRandom(Matrix* m, float min, float max)
{
	srand(time(0));
	for (size_t i = 0; i < m->GetRowCount() * m->GetColumnCount(); i++)
	{
		signed int r = rand() % ((int)(max * 1000) - (int)(min * 1000) + 1) + (int)(min * 1000);
		m->SetValue(i, r / 1000.0f);
	}
}

void MatrixMath::Copy(Matrix* from, Matrix* to)
{
	for (size_t i = 0; i < to->GetColumnCount() * to->GetRowCount(); i++)
		to->SetValue(i, from->GetValue(i));
}

void MatrixMath::AddIn(Matrix* a, Matrix* b)
{
	if (!SizeCheck(a, b))
		throw MatrixSizeException();
	for (unsigned int i = 0; i < a->GetColumnCount() * a->GetRowCount(); i++)
		a->AdjustValue(i, b->GetValue(i));
}

void MatrixMath::Add(Matrix* matrix, float value)
{
	for (unsigned int i = 0; i < matrix->GetColumnCount() * matrix->GetRowCount(); i++)
		matrix->AdjustValue(i, value);
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
	//Matrix* c = new Matrix(a->GetRowCount(), a->GetColumnCount());
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

void MatrixMath::SubstractIn(Matrix* a, Matrix* b)
{
	if (!SizeCheck(a, b))
		throw MatrixException(); //the two matrices doesn't have the same column/row count!
	for (size_t i = 0; i < a->GetRowCount() * a->GetColumnCount(); i++)
		a->AdjustValue(i, -b->GetValue(i));
}

void MatrixMath::MultiplyIn(Matrix* a, float b)
{
	for (size_t i = 0; i < a->GetColumnCount() * a->GetRowCount(); i++)
		a->SetValue(i, a->GetValue(i) * b);
}

Matrix* MatrixMath::Multiply(Matrix* a, float b)
{
	Matrix* c = new Matrix(a->GetRowCount(), a->GetColumnCount());
	for (size_t i = 0; i < a->GetColumnCount() * a->GetRowCount(); i++)
		c->SetValue(i, a->GetValue(i) * b);
	return c;
}

Matrix* MatrixMath::Multiply(Matrix* a, Matrix* b, Matrix* c)
{
#if USE_GPU
	return GPUMath::Multiplication(a, b, c);
#else
	if (!c)
		c = new Matrix(a->GetRowCount(), b->GetColumnCount());
	CacheVector col, row;
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
				//temp.vec = _mm_dp_ps(col.vec, row.vec, 0xff);
				__m128 mul = _mm_mul_ps(col.vec, row.vec);
				__m128 shuf = _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(2, 3, 0, 1));
				__m128 sums = _mm_add_ps(mul, shuf);
				shuf = _mm_movehl_ps(shuf, sums);
				sums = _mm_add_ss(sums, shuf);
				float result = _mm_cvtss_f32(sums);

				c->AdjustValue(ar, bc, result);
			}

			br += 4;
		}

	}
	return c;

#endif // USE_GPU
}

void MatrixMath::ElementviseMultiply(Matrix* a, Matrix* b)
{
#if DEBUG
	if (!SizeCheck(a, b))
		throw MatrixException();
#endif // DEBUG
	for (unsigned int i = 0; i < a->GetRowCount() * a->GetColumnCount(); i++)
		a->SetValue(i, a->GetValue(i) * b->GetValue(i));
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
		m->ReloadFromOther(&trans);
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
		value += a->GetValue(i) * b->GetValue(i);
	return value;
}

float MatrixMath::Sum(Matrix* m)
{
	float value = 0;
	for (size_t i = 0; i < m->GetColumnCount() * m->GetRowCount(); i++)
		value += m->GetValue(i);
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

Matrix* MatrixMath::Power(Matrix* original, unsigned int power)
{
	if (power <= 0)
		return nullptr;
	if (power == 1)
		return new Matrix(*original);

	Matrix* pow = new Matrix(*original);
	Matrix* tmp = nullptr;
	for (size_t i = 2; i < power; i++)
	{
		tmp = Multiply(pow, original);
		Copy(tmp, pow);
		delete tmp; //dangling pointer?
		//tmp = nullptr;
	}

	return pow;
}

Matrix* MatrixMath::Concat(Matrix* a, Matrix* b, unsigned int dimension, Matrix* c)
{
	if (dimension == 0) //concat on rows
	{
#ifdef DEBUG
		if (a->GetColumnCount() != b->GetColumnCount())
			throw MatrixException();
#endif // DEBUG
		if (!c)
			c = new Matrix(a->GetRowCount() + b->GetRowCount(), a->GetColumnCount());
		for (unsigned int row = 0; row < a->GetRowCount(); row++)
		{
			for (unsigned int col = 0; col < a->GetColumnCount(); col++)
			{
				c->SetValue(row, col, a->GetValue(row, col));
			}
		}

		for (unsigned int row = 0; row < b->GetRowCount(); row++)
		{
			for (unsigned int col = 0; col < a->GetColumnCount(); col++)
			{
				c->SetValue(a->GetRowCount() + row, col, b->GetValue(row, col));
			}
		}

		return c;
	}
	if (dimension == 1) //concat on rows
	{
#ifdef DEBUG
		if (a->GetRowCount() != b->GetRowCount())
			throw MatrixException();
#endif // DEBUG
		if (!c)
			c = new Matrix(a->GetRowCount(), a->GetColumnCount() + b->GetColumnCount());
		for (unsigned int row = 0; row < a->GetRowCount(); row++)
		{
			for (unsigned int col = 0; col < a->GetColumnCount(); col++)
			{
				c->SetValue(row, col, a->GetValue(row, col));
			}
		}

		for (unsigned int row = 0; row < a->GetRowCount(); row++)
		{
			for (unsigned int col = 0; col < b->GetColumnCount(); col++)
			{
				c->SetValue(row, col + a->GetColumnCount(), a->GetValue(row, col));
			}
		}

		return c;
	}
}
