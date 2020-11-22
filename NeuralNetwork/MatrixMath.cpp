#include "MatrixMath.h"
#include <iostream>
#include "Constants.h"
#include "MatrixException.hpp"

bool MatrixMath::SizeCheck(const std::shared_ptr<Matrix> a, const std::shared_ptr<Matrix> b)
{
	return (a->GetColumnCount() == b->GetColumnCount()) && (a->GetRowCount() == b->GetRowCount());
}

bool MatrixMath::IsVector(std::shared_ptr<Matrix> matrix)
{
	return (matrix->GetColumnCount() == 1 || matrix->GetRowCount() == 1);
}

bool MatrixMath::IsEqual(std::shared_ptr<Matrix> a, std::shared_ptr<Matrix> b)
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

void MatrixMath::FillWith(std::shared_ptr<Matrix> m, float value)
{
	for (unsigned int i = 0; i < m->GetColumnCount() * m->GetRowCount(); i++)
		m->SetValue(i, value);
}

void MatrixMath::FillWithRandom(std::shared_ptr<Matrix> m)
{

}

void MatrixMath::Copy(std::shared_ptr<Matrix> from, std::shared_ptr<Matrix> to)
{
	for (size_t i = 0; i < to->GetColumnCount() * to->GetRowCount(); i++)
		to->SetValue(i, (*from)[i]);
}

void MatrixMath::AddIn(std::shared_ptr<Matrix> a, std::shared_ptr<Matrix> b)
{
	if (!SizeCheck(a, b))
		throw MatrixSizeException();
	for (unsigned int i = 0; i < a->GetColumnCount() * a->GetRowCount(); i++)
		a->AdjustValue(i, (*b)[i]);
}

void MatrixMath::Add(std::shared_ptr<Matrix> matrix, float value)
{
	for (unsigned int i = 0; i < matrix->GetColumnCount() * matrix->GetRowCount(); i++)
		matrix->AdjustValue(i, value);
}

void Add(std::shared_ptr<Matrix> matrix, float value)
{
	for (size_t i = 0; i < matrix->GetColumnCount() * matrix->GetRowCount(); i++)
		matrix->AdjustValue(i, value);
}

std::shared_ptr<Matrix> MatrixMath::Add(const std::shared_ptr<Matrix> a, const std::shared_ptr<Matrix> b)
{
	if (!SizeCheck(a, b))
		throw MatrixException(); //the two matrices doesn't have the same column/row count!
	//std::shared_ptr<Matrix> c = new Matrix(a->GetRowCount(), a->GetColumnCount());
	std::shared_ptr<Matrix> c(new Matrix(a->GetRowCount(), a->GetColumnCount()));
	for (size_t i = 0; i < a->GetRowCount() * a->GetColumnCount(); i++)
		c->SetValue(i, a->GetValue(i) + b->GetValue(i));
	return c;
}

std::shared_ptr<Matrix> MatrixMath::Substract(std::shared_ptr<Matrix> a, std::shared_ptr<Matrix> b)
{
	if (!SizeCheck(a, b))
		throw MatrixException(); //the two matrices doesn't have the same column/row count!
	std::shared_ptr<Matrix> c(new Matrix(a->GetRowCount(), a->GetColumnCount()));
	for (size_t i = 0; i < a->GetRowCount() * a->GetColumnCount(); i++)
		c->SetValue(i, (*a)[i] - (*b)[i]);
	return c;
}

void MatrixMath::MultiplyIn(std::shared_ptr<Matrix> a, float b)
{
	for (size_t i = 0; i < a->GetColumnCount() * a->GetRowCount(); i++)
		a->SetValue(i, (*a)[i] * b);
}

std::shared_ptr<Matrix> MatrixMath::Multiply(std::shared_ptr<Matrix> a, float b)
{
	std::shared_ptr<Matrix> c(new Matrix(a->GetRowCount(), a->GetColumnCount()));
	for (size_t i = 0; i < a->GetColumnCount() * a->GetRowCount(); i++)
		c->SetValue(i, (*a)[i] * b);
	return c;
}

std::shared_ptr<Matrix> MatrixMath::Multiply(std::shared_ptr<Matrix> a, std::shared_ptr<Matrix> b, std::shared_ptr<Matrix> c)
{
	if (!c)
		std::shared_ptr<Matrix> c(new Matrix(a->GetRowCount(), a->GetColumnCount()));
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
}

void MatrixMath::ElementviseMultiply(std::shared_ptr<Matrix> a, std::shared_ptr<Matrix> b)
{
#if DEBUG
	if (!SizeCheck(a, b))
		throw MatrixException();
#endif // DEBUG
	for (unsigned int row = 0; row < a->GetRowCount(); row++)
	{
		for (unsigned int col = 0; col < a->GetColumnCount(); col++)
		{
			a->SetValue(row, col, a->GetValue(row, col) * b->GetValue(row, col));
		}
	}
}

std::shared_ptr<Matrix> MatrixMath::SlowMultiply(std::shared_ptr<Matrix> a, std::shared_ptr<Matrix> b)
{
	std::shared_ptr<Matrix> c(new Matrix(a->GetRowCount(), b->GetColumnCount()));
	MatrixMath::FillWith(c, 0);
	for (unsigned int i = 0; i < a->GetRowCount(); i++)
		for (unsigned int j = 0; j < b->GetColumnCount(); j++)
			for (unsigned int k = 0; k < a->GetColumnCount(); k++)
				c->AdjustValue(i, j, a->GetValue(i, k) * b->GetValue(k, j));
	return c;
}

void MatrixMath::Transpose(std::shared_ptr<Matrix> m)
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
		Copy(std::make_shared<Matrix>(trans), m);
	}
}

std::shared_ptr<Matrix> MatrixMath::GetRowMatrix(std::shared_ptr<Matrix> m, size_t row)
{
	std::shared_ptr<Matrix> rowm(new Matrix(1, m->GetColumnCount()));
	for (size_t i = 0; i < m->GetColumnCount(); i++)
		rowm->SetValue(i, m->GetValue(row, i));
	return rowm;
}

std::shared_ptr<Matrix> MatrixMath::GetColumnMatrix(std::shared_ptr<Matrix> m, size_t column)
{
	std::shared_ptr<Matrix> colm(new Matrix(m->GetRowCount(), 1));
	for (size_t i = 0; i < m->GetRowCount(); i++)
		colm->SetValue(i, m->GetValue(i, column));
	return colm;
}

std::shared_ptr<Matrix> MatrixMath::Hadamard(std::shared_ptr<Matrix> a, std::shared_ptr<Matrix> b)
{
	if (!SizeCheck(a, b))
		throw MatrixException(); //the two matrices doesn't have the same column/row count!
	std::shared_ptr<Matrix> cmat(new Matrix(a->GetRowCount(), a->GetColumnCount()));
	for (size_t r = 0; r < a->GetRowCount(); r++)
		for (size_t c = 0; c < a->GetColumnCount(); c++)
			cmat->SetValue(r, c, a->GetValue(r, c) * b->GetValue(r, c));
	return cmat;
}

std::shared_ptr<Matrix> MatrixMath::OuterProduct(std::shared_ptr<Matrix> a, std::shared_ptr<Matrix> b)
{
	if ((!IsVector(a) && !IsVector(b)) || (a->GetRowCount() == b->GetRowCount()))
		throw MatrixException(); //the two matrices must be vectors
	std::shared_ptr<Matrix> cmat;
	if (a->GetRowCount() == 1)
	{
		cmat.reset(new Matrix(b->GetRowCount(), a->GetColumnCount()));
		for (size_t r = 0; r < b->GetRowCount(); r++)
			for (size_t c = 0; c < a->GetColumnCount(); c++)
				cmat->SetValue(r, c, (*b)[r] * (*a)[c]);
	}
	else
	{
		cmat.reset(new Matrix(a->GetRowCount(), b->GetColumnCount()));
		for (size_t r = 0; r < a->GetRowCount(); r++)
			for (size_t c = 0; c < b->GetColumnCount(); c++)
				cmat->SetValue(r, c, (*a)[r] * (*b)[c]);
	}

	return cmat;
}

std::shared_ptr<Matrix> MatrixMath::CreateSubMatrix(std::shared_ptr<Matrix> m, size_t startRow, size_t startCol, size_t rowCount, size_t colCount)
{
	if (startRow + rowCount >= m->GetRowCount() || startCol + colCount >= m->GetColumnCount())
		throw MatrixException();
	std::shared_ptr<Matrix> mat(new Matrix(rowCount, colCount));
	for (size_t row = 0; row < rowCount; row++)
	{
		for (size_t col = 0; col < colCount; col++)
		{
			mat->SetValue(row, col, m->GetValue(startRow + row, startCol + col));
		}
	}

	return mat;
}

float MatrixMath::DotProduct(std::shared_ptr<Matrix> a, std::shared_ptr<Matrix> b)
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

float MatrixMath::Sum(std::shared_ptr<Matrix> m)
{
	float value = 0;
	for (size_t i = 0; i < m->GetColumnCount() * m->GetRowCount(); i++)
		value += (*m)[i];
	return value;
}

std::shared_ptr<Matrix> MatrixMath::Eye(unsigned int size)
{
	std::shared_ptr<Matrix> m(new Matrix(size, size));
	for (size_t i = 0; i < size; i++)
		m->SetValue(i, i, 1);
	return m;
}

void MatrixMath::PrintMatrix(std::shared_ptr<Matrix> m)
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

std::shared_ptr<Matrix> MatrixMath::Power(std::shared_ptr<Matrix> original, unsigned int power)
{
	std::shared_ptr<Matrix> pow(new Matrix(*original));

	return pow;
}
