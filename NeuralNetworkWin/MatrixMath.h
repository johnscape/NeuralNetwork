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
	/**
	 * @brief Checks if two matrices have the same dimensions.
	 * @param a The first matrix
	 * @param b The second matrix
	 * @return Returns true if the two matrices have the same row and column count
	*/
	inline bool SizeCheck(const Matrix& a, const Matrix& b);

	/**
	 * @brief Checks if a matrix have 1 as either dimension
	 * @param matrix The matrix to check
	 * @return Returns true, if either the row or column count equals 1
	*/
	inline bool IsVector(const Matrix& matrix);

	/**
	 * @brief Checks if two matrices are identical
	 * @param a The first matrix
	 * @param b The second matrix
	 * @return >Returns true, if the two matrices are the same size, and contains the same values
	*/
	bool IsEqual(const Matrix& a, const Matrix& b);

	/**
	 * @brief Fills the matrix with a specified value
	 * @param m The matrix to fill
	 * @param value The value to fill with
	*/
	void FillWith(Matrix& m, float value);

	/**
	 * @brief Fills the matrix with a random value.
	 * @param m The matrix to fill.
	 * @param min The minimum value of the random numbers.
	 * @param max The maximum value of the random numbers.
	*/
	void FillWithRandom(Matrix& m, float min = -1, float max = 1);

	/**
	 * @brief Copies one matrix into another. The two matrices must have the same size!
	 * @param from The matrix to copy from.
	 * @param to The matrix to copy to.
	*/
	void Copy(Matrix& from, Matrix& to);

	/**
	 * @brief Adds one matrix to the other: A += B
	 * @param a The matrix to store the new value
	 * @param b The matrix to add.
	*/
	void AddIn(Matrix& a, const Matrix& b);

	/**
	 * @brief Adds a constant value to every cell of the matrix.
	 * @param matrix The matrix
	 * @param value The value
	*/
	void Add(Matrix& matrix, float value);

	/**
	 * @brief Adds two matrix together and returns the new matrix: C = A + B
	 * @param a The first matrix
	 * @param b The second matrix
	 * @return The result of the operation
	*/
	Matrix Add(const Matrix& a, const Matrix& b);

	/**
	 * @brief Substracts two matrices from each other: C = A - B
	 * @param a The first matrix
	 * @param b The second matrix
	 * @return The result of the operation
	*/
	Matrix Substract(const Matrix& a, const Matrix& b);

	/**
	 * @brief Substact one matrix from the other: A-= B
	 * @param a The first matrix
	 * @param b The second matrix
	*/
	void SubstractIn(Matrix& a, const Matrix& b);

	/**
	 * @brief Multiplies the matrix with a value
	 * @param a The matrix
	 * @param b The value
	*/
	void MultiplyIn(Matrix& a, float b);

	/**
	 * @brief Multiplies the matrix with a value, then returns with it: C = A * b
	 * @param a The matrix
	 * @param b The value
	 * @return The new matrix
	*/
	Matrix Multiply(const Matrix& a, float b);

	/**
	 * @brief Multiplies two matrices together
	 * @param a The first matrix
	 * @param b The second matrix
	 * @return The result matrix.
	*/
	Matrix Multiply(const Matrix& a, const Matrix& b);

	void Multiply(const Matrix& a, const Matrix& b, Matrix& result);

	/**
	 * @brief Multiplies A's values with B's values. The two must have the same size!
	 * @param a The first matrix
	 * @param b The second matrix
	*/
	void ElementviseMultiply(Matrix& a, const Matrix& b);

	/**
	 * @brief Transposes a matrix
	 * @param m The matrix
	*/
	void Transpose(Matrix& m);

	/**
	 * @brief Creates a new row matrix from the original's specified row
	 * @param m The original matrix
	 * @param row The selected row
	 * @return A new row matrix
	*/
	Matrix GetRowMatrix(const Matrix& m, size_t row);

	/**
	 * @brief Creates a new column matrix from the original's specified column
	 * @param m The original matrix
	 * @param column The selected column
	 * @return A new column matrix
	*/
	Matrix GetColumnMatrix(const Matrix& m, size_t column);

	/**
	 * @brief Calculates the outer product of two matrices.
	 * @param a The first matrix
	 * @param b The second matrix
	 * @return The result of the outer product
	*/
	Matrix OuterProduct(const Matrix& a, const Matrix& b);

	/**
	 * @brief Calculates the dot product of two vectors
	 * @param a The first vector
	 * @param b The second vector
	 * @return The dot product
	*/
	float DotProduct(const Matrix& a, const Matrix& b);

	/**
	 * @brief Sums the values inside of a matrix
	 * @param m The matrix
	 * @return The sum of values inside of the matrix
	*/
	float Sum(const Matrix& m);

	/**
	 * @brief Creates an identity matrix of specified size
	 * @param size The size of the matrix
	 * @return An identity matrix of size: (size)x(size)
	*/
	Matrix Eye(unsigned int size);

	/**
	 * @brief Prints the matrix into the console
	 * @param m The matrix
	*/
	void PrintMatrix(const Matrix& m);

	/**
	 * @brief Calculates the power of a matrix
	 * @param original The matrix
	 * @param power The power
	 * @return The result of matrix^power
	*/
	Matrix Power(const Matrix& original, unsigned int power);

	/**
	 * @brief Concatenates two matrices
	 * @param a The first matrix
	 * @param b The second matrix
	 * @param dimension The dimension where the concatenation will happen (0 - concat at rows, 1 - concat at columns)
	 * @return The concatenated matrix
	*/
	Matrix Concat(const Matrix& a, const Matrix& b, unsigned int dimension);
}

#endif