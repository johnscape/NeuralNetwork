#include "Matrix.h"

#define BLOCK_SIZE 16

namespace GPUMath
{
	Matrix* Multiplication(Matrix* a, Matrix* b, Matrix* c);
	void ElementviseMultiply(Matrix* a, Matrix* b);
	void SubstractIn(Matrix* a, Matrix* b);
	void AddIn(Matrix* a, Matrix* b);

	void FillWith(Matrix* a, float value);
}