#include "Matrix.h"

namespace GPUMath
{
	unsigned int CalculateMaxBlockSize(Matrix* a, Matrix* b, unsigned int max);
	Matrix* Multiplication(Matrix* a, Matrix* b, Matrix* c);
	void AddIn(Matrix* a, Matrix* b);
}