#include "NeuralNetwork/MatrixMath.h"
#include <iostream>
#include "NeuralNetwork/Constants.h"
#include "NeuralNetwork/MatrixException.hpp"

#if USE_GPU
#include "MatrixGPUMath.cuh"
#endif

union Float2Int
{
	float f;
	int i;
};

bool MatrixMath::SizeCheck(const Matrix& a, const Matrix& b)
{
	return (a.GetColumnCount() == b.GetColumnCount()) && (a.GetRowCount() == b.GetRowCount());
}

bool MatrixMath::IsVector(const Matrix& matrix)
{
	return (matrix.GetColumnCount() == 1 || matrix.GetRowCount() == 1);
}

bool MatrixMath::IsEqual(const Matrix& a, const Matrix& b)
{
	if (!SizeCheck(a, b))
		return false;
	for (size_t i = 0; i < a.GetColumnCount() * a.GetRowCount(); i++)
	{
		if (a.GetValue(i) != b.GetValue(i))
			return false;
	}
	return true;
}

void MatrixMath::FillWith(Matrix& m, float value)
{
	for (unsigned int i = 0; i < m.GetColumnCount() * m.GetRowCount(); i++)
		m.SetValue(i, value);
#if USE_GPU
	GPUMath::FillWith(m, value);
#endif // USE_GPU

}

void MatrixMath::FillWithRandom(Matrix& m, float min, float max)
{
	srand(time(0));
	for (size_t i = 0; i < m.GetRowCount() * m.GetColumnCount(); i++)
	{
		signed int r = rand() % ((int)(max * 1000) - (int)(min * 1000) + 1) + (int)(min * 1000);
		m.SetValue(i, r / 1000.0f);
	}

#if USE_GPU
	m.CopyToGPU();
#endif // USE_GPU

}

void MatrixMath::Copy(const Matrix& from, Matrix& to)
{
#if USE_GPU
	from.CopyFromGPU();
#endif
	for (size_t i = 0; i < to.GetColumnCount() * to.GetRowCount(); i++)
		to.SetValue(i, from.GetValue(i));
#if USE_GPU
	to.CopyToGPU();
#endif // USE_GPU

}

void MatrixMath::AddIn(Matrix& a, const Matrix& b)
{
	if (!SizeCheck(a, b))
		throw MatrixSizeException();
#if USE_GPU
	GPUMath::AddIn(a, b);
#else
	for (unsigned int i = 0; i < a.GetColumnCount() * a.GetRowCount(); i++)
		a.AdjustValue(i, b.GetValue(i));
#endif
}

void MatrixMath::Add(Matrix& matrix, float value)
{
	for (unsigned int i = 0; i < matrix.GetColumnCount() * matrix.GetRowCount(); i++)
		matrix.AdjustValue(i, value);
}

Matrix MatrixMath::Add(const Matrix& a, const Matrix& b)
{
	if (!SizeCheck(a, b))
		throw MatrixException(); //the two matrices doesn't have the same column/row count!
	//Matrix* c = new Matrix(a->GetRowCount(), a->GetColumnCount());
	Matrix c(a.GetRowCount(), a.GetColumnCount());
	for (size_t i = 0; i < a.GetRowCount() * a.GetColumnCount(); i++)
		c.SetValue(i, a[i] + b[i]);
	return c;
}

Matrix MatrixMath::Substract(const Matrix& a, const Matrix& b)
{
	if (!SizeCheck(a, b))
		throw MatrixException(); //the two matrices doesn't have the same column/row count!
	Matrix c(a.GetRowCount(), a.GetColumnCount());
	for (size_t i = 0; i < a.GetRowCount() * a.GetColumnCount(); i++)
		c.SetValue(i, a[i] - b[i]);
	return c;
}

void MatrixMath::SubstractIn(Matrix& a, const Matrix& b)
{
	if (!SizeCheck(a, b))
		throw MatrixException(); //the two matrices doesn't have the same column/row count!
	for (size_t i = 0; i < a.GetRowCount() * a.GetColumnCount(); i++)
		a.AdjustValue(i, -b.GetValue(i));
}

void MatrixMath::MultiplyIn(Matrix& a, float b)
{
	for (size_t i = 0; i < a.GetColumnCount() * a.GetRowCount(); i++)
		a.SetValue(i, a.GetValue(i) * b);
}

Matrix MatrixMath::Multiply(const Matrix& a, float b)
{
	Matrix c(a.GetRowCount(), a.GetColumnCount());
	for (size_t i = 0; i < a.GetColumnCount() * a.GetRowCount(); i++)
		c.SetValue(i, a[i] * b);
	return c;
}

Matrix MatrixMath::Multiply(const Matrix& a, const Matrix& b)
{
	Matrix result(a.GetRowCount(), b.GetColumnCount());
	Multiply(a, b, result);
	return result;
}

void MatrixMath::Multiply(const Matrix& a, const Matrix& b, Matrix& result)
{
#if USE_GPU
	return GPUMath::Multiplication(a, b, result);
#else
	CacheVector col, row;
	size_t br;
	for (size_t bc = 0; bc < b.GetColumnCount(); bc++)
	{
		br = 0;
		while (br < b.GetRowCount())
		{
			for (unsigned char i = 0; i < 4; i++)
				col.fl[i] = b.GetValue(br + i, bc);
			for (size_t ar = 0; ar < a.GetRowCount(); ar++)
			{
				for (size_t i = 0; i < 4; i++)
					row.fl[i] = a.GetValue(ar, br + i);
				//temp.vec = _mm_dp_ps(col.vec, row.vec, 0xff);
				__m128 mul = _mm_mul_ps(col.vec, row.vec);
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
#endif // USE_GPU
}

void MatrixMath::ElementviseMultiply(Matrix& a, const Matrix& b)
{
#if DEBUG
	if (!SizeCheck(a, b))
		throw MatrixException();
#endif // DEBUG
#if USE_GPU
	GPUMath::ElementviseMultiply(a, b);
#else
	for (unsigned int i = 0; i < a.GetRowCount() * a.GetColumnCount(); i++)
		a.SetValue(i, a.GetValue(i) * b.GetValue(i));
#endif // USE_GPU
}

void MatrixMath::Transpose(Matrix& m)
{
	if (m.GetColumnCount() == m.GetRowCount())
	{
		size_t count = m.GetColumnCount();
		for (size_t r = 0; r < count; r++)
		{
			for (size_t c = 0; c < count; c++)
			{
				if (r >= c)
					continue;
				float temp = m.GetValue(r, c);
				m.SetValue(r, c, m.GetValue(c, r));
				m.SetValue(c, r, temp);
			}
		}
	}
	else
	{
		Matrix trans(m.GetColumnCount(), m.GetRowCount());
		for (unsigned int i = 0; i < m.GetRowCount(); i++)
			for (unsigned int j = 0; j < m.GetColumnCount(); j++)
				trans.SetValue(j, i, m.GetValue(i, j));
		m.ReloadFromOther(trans);
	}
}

Matrix MatrixMath::GetRowMatrix(const Matrix& m, size_t row)
{
	Matrix rowm(1, m.GetColumnCount());
	for (size_t i = 0; i < m.GetColumnCount(); i++)
		rowm.SetValue(i, m.GetValue(row, i));
	return rowm;
}

Matrix MatrixMath::GetColumnMatrix(const Matrix& m, size_t column)
{
	Matrix colm(m.GetRowCount(), 1);
	for (size_t i = 0; i < m.GetRowCount(); i++)
		colm.SetValue(i, m.GetValue(i, column));
	return colm;
}

Matrix MatrixMath::OuterProduct(const Matrix& a, const Matrix& b)
{
	if ((!IsVector(a) && !IsVector(b)) || (a.GetRowCount() == b.GetRowCount()))
		throw MatrixException(); //the two matrices must be vectors
	Matrix cmat;
	if (a.GetRowCount() == 1)
	{
		cmat = Matrix(b.GetRowCount(), a.GetColumnCount());
		for (size_t r = 0; r < b.GetRowCount(); r++)
			for (size_t c = 0; c < a.GetColumnCount(); c++)
				cmat.SetValue(r, c, b[r] * a[c]);
	}
	else
	{
		cmat = Matrix(a.GetRowCount(), b.GetColumnCount());
		for (size_t r = 0; r < a.GetRowCount(); r++)
			for (size_t c = 0; c < b.GetColumnCount(); c++)
				cmat.SetValue(r, c, a[r] * b[c]);
	}

	return cmat;
}

float MatrixMath::DotProduct(const Matrix& a, const Matrix& b)
{
	if ((!IsVector(a) && !IsVector(b)))
		throw MatrixException(); //the two matrices must be vectors
	if (b.GetColumnCount() * b.GetRowCount() != a.GetColumnCount() * a.GetRowCount())
		throw MatrixException();

	float value = 0;
	for (size_t i = 0; i < a.GetColumnCount() * a.GetRowCount(); i++)
		value += a.GetValue(i) * b.GetValue(i);
	return value;
}

float MatrixMath::Sum(const Matrix& m)
{
	float value = 0;
	for (size_t i = 0; i < m.GetColumnCount() * m.GetRowCount(); i++)
		value += m.GetValue(i);
	return value;
}

Matrix MatrixMath::Eye(unsigned int size)
{
	Matrix m(size, size);
	for (size_t i = 0; i < size; i++)
		m.SetValue(i, i, 1);
	return m;
}

void MatrixMath::PrintMatrix(const Matrix& m)
{
	for (unsigned int r = 0; r < m.GetRowCount(); r++)
	{
		for (unsigned int c = 0; c < m.GetColumnCount(); c++)
		{
			std::cout << m.GetValue(r, c) << " ";
		}
		std::cout << std::endl;
	}
}

Matrix MatrixMath::Power(const Matrix& original, unsigned int power)
{
	if (power <= 0)
		return Matrix();
	if (power == 1)
		return Matrix(original);

	Matrix pow(original);
	Matrix tmp;
	for (size_t i = 2; i < power; i++)
	{
		tmp = Multiply(pow, original);
#if USE_GPU
		tmp.CopyFromGPU();
#endif // USE_GPU

		Copy(tmp, pow);
	}

	return pow;
}

Matrix MatrixMath::Concat(const Matrix& a, const Matrix& b, unsigned int dimension) //TODO: move to matrix
{
	if (dimension == 0) //concat on rows
	{
#ifdef DEBUG
		if (a.GetColumnCount() != b.GetColumnCount())
			throw MatrixException();
#endif // DEBUG
		Matrix c(a.GetRowCount() + b.GetRowCount(), a.GetColumnCount());
		for (unsigned int row = 0; row < a.GetRowCount(); row++)
		{
			for (unsigned int col = 0; col < a.GetColumnCount(); col++)
			{
				c.SetValue(row, col, a.GetValue(row, col));
			}
		}

		for (unsigned int row = 0; row < b.GetRowCount(); row++)
		{
			for (unsigned int col = 0; col < a.GetColumnCount(); col++)
			{
				c.SetValue(a.GetRowCount() + row, col, b.GetValue(row, col));
			}
		}

		return c;
	}
	if (dimension == 1) //concat on cols
	{
#ifdef DEBUG
		if (a.GetRowCount() != b.GetRowCount())
			throw MatrixException();
#endif // DEBUG
		Matrix c(a.GetRowCount(), a.GetColumnCount() + b.GetColumnCount());
		for (size_t row = 0; row < c.GetRowCount(); row++)
		{
			for (size_t col = 0; col < c.GetColumnCount(); col++)
			{
				float val = 0;
				if (col < a.GetColumnCount())
					val = a.GetValue(row, col);
				else
					val = b.GetValue(row, col - a.GetColumnCount());
				c.SetValue(row, col, val);
			}
		}

		return c;
	}
}
