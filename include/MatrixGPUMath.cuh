#include "Matrix.h"

#define BLOCK_SIZE 16

namespace GPUMath
{
	Matrix& Multiplication(const Matrix& a, const Matrix& b);
	void Multiplication(const Matrix& a, const Matrix& b, Matrix& c);
	void ElementviseMultiply(Matrix& a, const Matrix& b);
	void SubstractIn(Matrix& a, const Matrix& b);
	void AddIn(Matrix& a, const Matrix& b);

	void FillWith(Matrix& a, float value);
}