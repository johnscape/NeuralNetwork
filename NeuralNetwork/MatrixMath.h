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
	inline bool SizeCheck(const std::shared_ptr<Matrix> a, const std::shared_ptr<Matrix> b);

	/// <summary>
	/// Checks if a matrix have 1 as either dimension
	/// </summary>
	/// <param name="matrix">The matrix to check</param>
	/// <returns>Returns true, if either the row or column count equals 1</returns>
	inline bool IsVector(std::shared_ptr<Matrix> matrix);

	/// <summary>
	/// Checks if two matrices are identical
	/// </summary>
	/// <param name="a">The first matrix</param>
	/// <param name="b">The second matrix</param>
	/// <returns>Returns true, if the two matrices are the same size, and contains the same values</returns>
	bool IsEqual(std::shared_ptr<Matrix> a, std::shared_ptr<Matrix> b);


	/// <summary>
	/// Fills the matrix with a specified value
	/// </summary>
	/// <param name="m">The matrix to fill</param>
	/// <param name="value">The value</param>
	void FillWith(std::shared_ptr<Matrix> m, float value);

	void FillWithRandom(std::shared_ptr<Matrix> m);

	void Copy(std::shared_ptr<Matrix> from, std::shared_ptr<Matrix> to);
	void AddIn(std::shared_ptr<Matrix> a, std::shared_ptr<Matrix> b);
	void Add(std::shared_ptr<Matrix> matrix, float value);
	std::shared_ptr<Matrix> Add(const std::shared_ptr<Matrix> a, const std::shared_ptr<Matrix> b);
	std::shared_ptr<Matrix> Substract(std::shared_ptr<Matrix> a, std::shared_ptr<Matrix> b);
	void MultiplyIn(std::shared_ptr<Matrix> a, float b);
	std::shared_ptr<Matrix> Multiply(std::shared_ptr<Matrix> a, float b);
	std::shared_ptr<Matrix> Multiply(std::shared_ptr<Matrix> a, std::shared_ptr<Matrix> b, std::shared_ptr<Matrix> c = nullptr);
	void ElementviseMultiply(std::shared_ptr<Matrix> a, std::shared_ptr<Matrix> b);
	std::shared_ptr<Matrix> SlowMultiply(std::shared_ptr<Matrix> a, std::shared_ptr<Matrix> b);
	void Transpose(std::shared_ptr<Matrix> m);
	std::shared_ptr<Matrix> GetRowMatrix(std::shared_ptr<Matrix> m, size_t row);
	std::shared_ptr<Matrix> GetColumnMatrix(std::shared_ptr<Matrix> m, size_t column);
	std::shared_ptr<Matrix> Hadamard(std::shared_ptr<Matrix> a, std::shared_ptr<Matrix> b);
	std::shared_ptr<Matrix> OuterProduct(std::shared_ptr<Matrix> a, std::shared_ptr<Matrix> b);
	std::shared_ptr<Matrix> CreateSubMatrix(std::shared_ptr<Matrix> m, size_t startRow, size_t startCol, size_t rowCount, size_t colCount);
	float DotProduct(std::shared_ptr<Matrix> a, std::shared_ptr<Matrix> b);
	float Sum(std::shared_ptr<Matrix> m);
	std::shared_ptr<Matrix> Eye(unsigned int size);

	void PrintMatrix(std::shared_ptr<Matrix> m);

	std::shared_ptr<Matrix> Power(std::shared_ptr<Matrix> original, unsigned int power);
}

#endif // !MATRIXMATH_GUARD