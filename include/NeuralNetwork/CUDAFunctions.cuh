#pragma once
#include "Matrix.h"
#include "Tensor.h"

/**
 * @brief A namespace containing CUDA functions for matrix and tensor operations
 * 
 */
namespace MatrixCUDAMath
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

namespace TensorCUDAMath
{
    //Addition
    /**
     * Adds two tensors together, stores the result in the third (c = a + b)
     * @param a The first tensor to add
     * @param b The second tensor to add
     * @param c The tensor to store the results in
     */
    void Add(const Tensor& a, const Tensor& b, Tensor& c);

    /**
     * Adds the second tensor to the first one (a += b)
     * @param a The tensor to add to
     * @param b The tensor to add
     */
    void AddIn(Tensor& a, const Tensor& b);

    /**
     * Adds a matrix to the tensor
     * @param a The tensor to add to
     * @param b The matrix to add
     */
    void AddIn(Tensor& a, const Matrix& b);

    /**
     * Adds a constant value to each element of the tensor
     * @param a The tensor to increment
     * @param v The value to add
     */
    void AddConstant(Tensor& a, float v);

    //Subtraction
    /**
     * Subtract two tensors from each other, stores the result in the third one (c = a - b)
     * @param a The tensor to subtract from
     * @param b The tensor to subtract
     * @param c The tensor with the results
     */
    void Subtract(const Tensor& a, const Tensor& b, Tensor& c);

    /**
     * Subtracts the second tensor from the first one (a -= b)
     * @param a The tensor to subtract from
     * @param b The tensor to subtract
     */
    void SubtractIn(Tensor& a, const Tensor& b);

    /**
     * Subtracts a matrix from the tensor
     * @param a The tensor
     * @param b The matrix
     */
    void SubtractIn(Tensor& a, const Matrix& b);

    /**
     * Subtract a value from each element of the tensor
     * @param a The tensor to subtract from
     * @param v The value to subtract
     */
    void SubtractConstant(Tensor& a, float v);

    //Multiplication
    /**
     * Multiplies two tensors together, stores the result in the third one (c = a * b)
     * @param a The first tensor to multiply
     * @param b The second tensor to multiply
     * @param c The result tensor
     */
    void Multiplication(const Tensor& a, const Tensor& b, Tensor& c);

    /**
     * Multiplies a tensor and a matrix together.
     * @param a The tensor to multiply
     * @param b The matrix to multiply
     * @param c The result tensor
     */
    void Multiplication(const Tensor& a, const Matrix& b, Tensor& c);

    /**
     * Multiplies two tensors elementwise. Stores the result in the first one
     * @param a The first tensor to multiply, stores the result
     * @param b The second tensor to multiply
     */
    void ElementwiseMultiply(Tensor& a, const Tensor& b);

    /**
     * Multiplies a tensor with a single value, elementwise.
     * @param a The tensor to multiply
     * @param v The value to multiply with
     */
    void MultiplyConstant(Tensor& a, float v);

    //Misc

    /**
     * @brief Fills a tensor CUDA values with a fixed value.
     *
     * @param a The tensor to fill
     * @param value The value to fill with
     */
    void FillWith(Tensor& a, float value);
}

namespace CUDAOperations
{
    /**
     * Copies part of the GPU values from an origin to a target.
     * @param target The target to copy to and override
     * @param origin The origin to copy from
     * @param targetOffset The offset of the target, where the overriding should start
     * @param originOffset The offset of the origin, where the values will be copied from
     * @param count The count of elements to copy
     */
    void CopyPartTo(Matrix& target, const Matrix& origin, unsigned int targetOffset, unsigned int originOffset, unsigned int count);

    /**
     * Copies part of the GPU values from an origin to a target.
     * @param target The target to copy to and override
     * @param origin The origin to copy from
     * @param targetOffset The offset of the target, where the overriding should start
     * @param originOffset The offset of the origin, where the values will be copied from
     * @param count The count of elements to copy
     */
    void CopyPartTo(Tensor& target, const Tensor& origin, unsigned int targetOffset, unsigned int originOffset, unsigned int count);
}