#include "Matrix.h"

#define BLOCK_SIZE 16

/**
 * @brief A namespace containing CUDA functions for matrix and tensor operations
 * 
 */
namespace GPUMath
{
	//Addition
	Matrix& Add(const Matrix& a, const Matrix& b);
	void Add(const Matrix& a, const Matrix& b, Matrix& c);
	void AddIn(Matrix& a, const Matrix& b);

	Matrix& AddConstant(const Matrix& a, float v);
	void AddConstant(Matrix& a, float v);

	//Subtraction
	Matrix& Subtract(const Matrix& a, const Matrix& b);
	void Subtract(const Matrix& a, const Matrix& b, Matrix& c);
	void SubtractIn(Matrix& a, const Matrix& b);

	Matrix& SubtractConstant(const Matrix& a, float v);
	void SubtractConstant(Matrix& a, float v);

	//Multiplication
	Matrix& Multiplication(const Matrix& a, const Matrix& b);
	void Multiplication(const Matrix& a, const Matrix& b, Matrix& c);

	void ElementviseMultiply(Matrix& a, const Matrix& b);

	Matrix& MultiplyConstant(const Matrix& a, float v);
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