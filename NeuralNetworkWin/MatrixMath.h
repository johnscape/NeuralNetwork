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
	inline bool SizeCheck(const Matrix* a, const Matrix* b);

	/**
	 * @brief Checks if a matrix have 1 as either dimension
	 * @param matrix The matrix to check
	 * @return Returns true, if either the row or column count equals 1
	*/
	inline bool IsVector(const Matrix* matrix);

	/**
	 * @brief Checks if two matrices are identical
	 * @param a The first matrix
	 * @param b The second matrix
	 * @return >Returns true, if the two matrices are the same size, and contains the same values
	*/
	bool IsEqual(const Matrix* a, const Matrix* b);

	/**
	 * @brief Fills the matrix with a specified value
	 * @param m The matrix to fill
	 * @param value The value to fill with
	*/
	void FillWith(Matrix* m, float value);

	/**
	 * @brief Fills the matrix with a random value.
	 * @param m The matrix to fill.
	 * @param min The minimum value of the random numbers.
	 * @param max The maximum value of the random numbers.
	*/
	void FillWithRandom(Matrix* m, float min = -1, float max = 1);

	/**
	 * @brief Copies one matrix into another. The two matrices must have the same size!
	 * @param from The matrix to copy from.
	 * @param to The matrix to copy to.
	*/
	void Copy(Matrix* from, Matrix* to);

	/**
	 * @brief Adds one matrix to the other: A += B
	 * @param a The matrix to store the new value
	 * @param b The matrix to add.
	*/
	void AddIn(Matrix* a, Matrix* b);

	/**
	 * @brief Adds a constant value to every cell of the matrix.
	 * @param matrix The matrix
	 * @param value The value
	*/
	void Add(Matrix* matrix, float value);

	/**
	 * @brief Adds two matrix together and returns the new matrix: C = A + B
	 * @param a The first matrix
	 * @param b The second matrix
	 * @return The result of the operation
	*/
	Matrix* Add(const Matrix* a, const Matrix* b);

	/**
	 * @brief Substracts two matrices from each other: C = A - B
	 * @param a The first matrix
	 * @param b The second matrix
	 * @return The result of the operation
	*/
	Matrix* Substract(Matrix* a, Matrix* b);

	/**
	 * @brief Substact one matrix from the other: A-= B
	 * @param a The first matrix
	 * @param b The second matrix
	*/
	void SubstractIn(Matrix* a, Matrix* b);

	/**
	 * @brief Multiplies the matrix with a value
	 * @param a The matrix
	 * @param b The value
	*/
	void MultiplyIn(Matrix* a, float b);

	/**
	 * @brief Multiplies the matrix with a value, then returns with it: C = A * b
	 * @param a The matrix
	 * @param b The value
	 * @return The new matrix
	*/
	Matrix* Multiply(Matrix* a, float b);

	/**
	 * @brief Multiplies two matrices together
	 * @param a The first matrix
	 * @param b The second matrix
	 * @param c Optional, if you have a matrix to add the result into, pass it here: C += A * B
	 * @return The result matrix.
	*/
	Matrix* Multiply(Matrix* a, Matrix* b, Matrix* c = nullptr);

	/**
	 * @brief Multiplies A's values with B's values. The two must have the same size!
	 * @param a The first matrix
	 * @param b The second matrix
	*/
	void ElementviseMultiply(Matrix* a, Matrix* b);

	/**
	 * @brief Multiplies two matrices together
	 * @param a The first matrix
	 * @param b The second matrix
	 * @return The result matrix.
	*/
	[[deprecated("Do not use, use Multiply instead!")]]
	Matrix* SlowMultiply(Matrix* a, Matrix* b);

	/**
	 * @brief Transposes a matrix
	 * @param m The matrix
	*/
	void Transpose(Matrix* m);

	/**
	 * @brief Creates a new row matrix from the original's specified row
	 * @param m The original matrix
	 * @param row The selected row
	 * @return A new row matrix
	*/
	Matrix* GetRowMatrix(Matrix* m, size_t row);

	/**
	 * @brief Creates a new column matrix from the original's specified column
	 * @param m The original matrix
	 * @param column The selected column
	 * @return A new column matrix
	*/
	Matrix* GetColumnMatrix(Matrix* m, size_t column);

	/**
	 * @brief Calculates the inner product of two matrices.
	 * @param a The first matrix
	 * @param b The second matrix
	 * @return The result of the inner product
	*/
	[[deprecated("Will be removed in future version")]]
	Matrix* Hadamard(Matrix* a, Matrix* b);

	/**
	 * @brief Calculates the outer product of two matrices.
	 * @param a The first matrix
	 * @param b The second matrix
	 * @return The result of the outer product
	*/
	Matrix* OuterProduct(Matrix* a, Matrix* b);

	/**
	 * @brief Creates a sub matrix from the original
	 * @param m The original matrix
	 * @param startRow The first row to select
	 * @param startCol The first column to select
	 * @param rowCount The number of selected rows
	 * @param colCount The number of selected columns
	 * @return A new matrix of the size: (rowCount - startRow)x(colCount - startCol)
	*/
	Matrix* CreateSubMatrix(Matrix* m, size_t startRow, size_t startCol, size_t rowCount, size_t colCount);

	/**
	 * @brief Calculates the dot product of two vectors
	 * @param a The first vector
	 * @param b The second vector
	 * @return The dot product
	*/
	float DotProduct(Matrix* a, Matrix* b);

	/**
	 * @brief Sums the values inside of a matrix
	 * @param m The matrix
	 * @return The sum of values inside of the matrix
	*/
	float Sum(Matrix* m);

	/**
	 * @brief Creates an identity matrix of specified size
	 * @param size The size of the matrix
	 * @return An identity matrix of size: (size)x(size)
	*/
	Matrix* Eye(unsigned int size);

	/**
	 * @brief Prints the matrix into the console
	 * @param m The matrix
	*/
	void PrintMatrix(Matrix* m);

	/**
	 * @brief Calculates the power of a matrix
	 * @param original The matrix
	 * @param power The power
	 * @return The result of matrix^power
	*/
	Matrix* Power(Matrix* original, unsigned int power);

	/**
	 * @brief Concatenates two matrices
	 * @param a The first matrix
	 * @param b The second matrix
	 * @param dimension The dimension where the concatenation will happen (0 - concat at rows, 1 - concat at columns)
	 * @param c Optional, the result will be stored here
	 * @return The concatenated matrix
	*/
	Matrix* Concat(Matrix* a, Matrix* b, unsigned int dimension, Matrix* c = nullptr);
}

#endif // !MATRIXMATH_GUARD