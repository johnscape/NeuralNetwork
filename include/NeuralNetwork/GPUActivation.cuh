#include "Matrix.h"
#include "Tensor.h"

namespace GPUActivation
{
    // Matrices

	///Sigmoid
	Matrix SigmoidCalculate(const Matrix& original);
	void SigmoidCalculate(const Matrix& from, Matrix& to);

	Matrix SigmoidInvCalculate(const Matrix& original);
	void SigmoidInvCalculate(const Matrix& from, Matrix& to);

	/// Tanh
	Matrix TanhCalculate(const Matrix& original);
	void TanhCalculate(const Matrix& from, Matrix& to);

	Matrix TanhInvCalculate(const Matrix& original);
	void TanhInvCalculate(const Matrix& from, Matrix& to);

    /// Softmax

    Matrix SoftmaxCalculate(const Matrix& original);
    void SoftmaxCalculate(const Matrix& from, Matrix& to);

    Matrix SoftmaxInvCalculate(const Matrix& original);
    void SoftmaxInvCalculate(const Matrix& from, Matrix& to);

    // Tensors

    ///Sigmoid
    Tensor SigmoidCalculate(const Tensor& original);
    void SigmoidCalculate(const Tensor& from, Tensor& to);

    Tensor SigmoidInvCalculate(const Tensor& original);
    void SigmoidInvCalculate(const Tensor& from, Tensor& to);

    /// Tanh
    Tensor TanhCalculate(const Tensor& original);
    void TanhCalculate(const Tensor& from, Tensor& to);

    Tensor TanhInvCalculate(const Tensor& original);
    void TanhInvCalculate(const Tensor& from, Tensor& to);

    /// Softmax
    Tensor SoftmaxCalculate(const Tensor& original);
    void SoftmaxCalculate(const Tensor& from, Tensor& to);

    Tensor SoftmaxInvCalculate(const Tensor& original);
    void SoftmaxInvCalculate(const Tensor& from, Tensor& to);

}
