#pragma once

#ifndef MATRIXMATH_GUARD
#define MATRIXMATH_GUARD

#include "Matrix.h"
#include "MatrixException.hpp"

#include <memory>

#include "nmmintrin.h"
#include "immintrin.h"

union CacheVector
{
	__m128 vec;
	float fl[4];
};

namespace MatrixMath
{
	/// <summary>
	/// Checks if two matrices have the same dimensions.
	/// </summary>
	/// <param name="a">The first matrix</param>
	/// <param name="b">The second matrix</param>
	/// <returns>Returns true if the two matrices have the same row and column count</returns>
	inline bool SizeCheck(const Matrix* a, const Matrix* b);

	/// <summary>
	/// Checks if a matrix have 1 as either dimension
	/// </summary>
	/// <param name="matrix">The matrix to check</param>
	/// <returns>Returns true, if either the row or column count equals 1</returns>
	inline bool IsVector(const Matrix* matrix);

	/// <summary>
	/// Checks if two matrices are identical
	/// </summary>
	/// <param name="a">The first matrix</param>
	/// <param name="b">The second matrix</param>
	/// <returns>Returns true, if the two matrices are the same size, and contains the same values</returns>
	bool IsEqual(const Matrix* a, const Matrix* b);


	/// <summary>
	/// Fills the matrix with a specified value
	/// </summary>
	/// <param name="m">The matrix to fill</param>
	/// <param name="value">The value</param>
	void FillWith(Matrix* m, float value);

	void FillWithRandom(Matrix* m, float min = -1, float max = 1);

	void Copy(Matrix* from, Matrix* to);
	void AddIn(Matrix* a, Matrix* b);
	void Add(Matrix* matrix, float value);
	Matrix* Add(const Matrix* a, const Matrix* b);
	Matrix* Substract(Matrix* a, Matrix* b);
	void SubstractIn(Matrix* a, Matrix* b);
	void MultiplyIn(Matrix* a, float b);
	Matrix* Multiply(Matrix* a, float b);
	Matrix* Multiply(Matrix* a, Matrix* b, Matrix* c = nullptr);
	void ElementviseMultiply(Matrix* a, Matrix* b);
	Matrix* SlowMultiply(Matrix* a, Matrix* b);
	void Transpose(Matrix* m);
	Matrix* GetRowMatrix(Matrix* m, size_t row);
	Matrix* GetColumnMatrix(Matrix* m, size_t column);
	Matrix* Hadamard(Matrix* a, Matrix* b);
	Matrix* OuterProduct(Matrix* a, Matrix* b);
	Matrix* CreateSubMatrix(Matrix* m, size_t startRow, size_t startCol, size_t rowCount, size_t colCount);
	float DotProduct(Matrix* a, Matrix* b);
	float Sum(Matrix* m);
	Matrix* Eye(unsigned int size);

	void PrintMatrix(Matrix* m);

	Matrix* Power(Matrix* original, unsigned int power);
	Matrix* Concat(Matrix* a, Matrix* b, unsigned int dimension, Matrix* c);
}

#endif // !MATRIXMATH_GUARD