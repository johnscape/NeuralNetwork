#include "Matrix.h"

#define BLOCK_SIZE 16

/**
 * @brief A namespace containing CUDA functions for matrix and tensor operations
 * 
 */
namespace GPUMath
{
	//Addition
    /**
     * Adds two matrices together, stores the result in the third (c = a + b)
     * @param a The first matrix to add
     * @param b The second matrix to add
     * @param c The matrix to store the results in
     */
	void Add(const Matrix& a, const Matrix& b, Matrix& c);

    /**
     * Adds the second matrix to the first one (a += b)
     * @param a The matrix to add to
     * @param b The matrix to add
     */
	void AddIn(Matrix& a, const Matrix& b);

    /**
     * Adds a constant value to each element of the matrix
     * @param a The matrix to increment
     * @param v The value to add
     */
	void AddConstant(Matrix& a, float v);

	//Subtraction
    /**
     * Subtract two matrices from each other, stores the result in the third one (c = a - b)
     * @param a The matrix to subtract from
     * @param b The matrix to subtract
     * @param c The matrix with the results
     */
	void Subtract(const Matrix& a, const Matrix& b, Matrix& c);

    /**
     * Subtracts the second matrix from the first one (a -= b)
     * @param a The matrix to subtract from
     * @param b The matrix to subtract
     */
	void SubtractIn(Matrix& a, const Matrix& b);

    /**
     * Subtract a value from each element of the matrix
     * @param a The matrix to subtract from
     * @param v The value to subtract
     */
	void SubtractConstant(Matrix& a, float v);

	//Multiplication
    /**
     * Multiplies two matrices together, stores the result in the third one (c = a * b)
     * @param a The first matrix to multiply
     * @param b The second matrix to multiply
     * @param c The result matrix
     */
	void Multiplication(const Matrix& a, const Matrix& b, Matrix& c);

    /**
     * Multiplies two matrices elementwise. Stores the result in the first one
     * @param a The first matrix to multiply, stores the result
     * @param b The second matrix to multiply
     */
	void ElementwiseMultiply(Matrix& a, const Matrix& b);

    /**
     * Multiplies a matrix with a single value, elementwise.
     * @param a The matrix to multiply
     * @param v The value to multiply with
     */
	void MultiplyConstant(Matrix& a, float v);

	//Misc

	/**
	 * @brief Fills a matrix CUDA values with a fixed value.
	 * 
	 * @param a The matrix to fill
	 * @param value The value to fill with
	 */
	void FillWith(Matrix& a, float value);

	
}