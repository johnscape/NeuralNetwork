#include "Matrix.h"

namespace GPUActivation
{
	///Sigmoid
	Matrix* SigmoidCalculate(Matrix* original);
	void SigmoidCalculate(Matrix* from, Matrix* to);

	Matrix* SigmoidInvCalculate(Matrix* original);
	void SigmoidInvCalculate(Matrix* from, Matrix* to);

	/// Tanh
	Matrix* TanhCalculate(Matrix* original);
	void TanhCalculate(Matrix* from, Matrix* to);

	Matrix* TanhInvCalculate(Matrix* original);
	void TanhInvCalculate(Matrix* from, Matrix* to);
}
