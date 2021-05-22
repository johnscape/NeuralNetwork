#include "Matrix.h"

namespace GPUActivation
{
	///Sigmoid
	Matrix& SigmoidCalculate(const Matrix& original);
	void SigmoidCalculate(const Matrix& from, Matrix& to);

	Matrix& SigmoidInvCalculate(const Matrix& original);
	void SigmoidInvCalculate(const Matrix& from, Matrix& to);

	/// Tanh
	Matrix& TanhCalculate(const Matrix& original);
	void TanhCalculate(const Matrix& from, Matrix& to);

	Matrix& TanhInvCalculate(const Matrix& original);
	void TanhInvCalculate(const Matrix& from, Matrix& to);
}
