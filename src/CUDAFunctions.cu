#include <cuda_runtime.h>
#include "NeuralNetwork/CUDAFunctions.cuh"
#include "NeuralNetwork/Constants.h"
#if USE_CUBLAS
#include <cublas_v2.h>
#endif

// Kernels

// A: aRows x aCols -
// B: aCols x bCols - x m
// C: aRows x bCols -
__global__ void MultiplicationKernel(const float* A, const float* B, float* C, int aRows, int aCols, int bCols)
{
	const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col < bCols && row < aRows)
	{
        float sum = 0;
		for (int i = 0; i < aCols; i++)
		{
			sum += A[row * aCols + i] * B[i * bCols + col];
		}
		C[row * bCols + col] = sum;
	}
}

__global__ void TensorMatMultiplicationKernel(const float* tensor, const float* matrix, float* result, int tensorRows, int tensorCols, int matrixCols, int matrixCount)
{
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int originalTensorOffset = tensorCols * tensorRows;
    const unsigned int resultTensorOffset = matrixCols * tensorRows;

    for (unsigned int m = 0; m < matrixCount; m++)
    {
        if (col < matrixCols && row < tensorRows)
        {
            float sum = 0;
            for (int i = 0; i < tensorCols; i++)
            {
                sum += tensor[row * tensorCols + i + m * originalTensorOffset] * matrix[i * matrixCols + col];
            }
            result[row * matrixCols + col + m * resultTensorOffset] = sum;
        }
    }
}

__global__ void AdditionKernel(float* A, const float* B, unsigned int maxNum)
{
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < maxNum)
		A[i] += B[i];
}

__global__ void TensorMatAdditionKernel(float* tensor, const float* matrix, unsigned int tensorSize, unsigned int matrixSize)
{
    const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < tensorSize)
        tensor[i] += matrix[i % matrixSize];
}

__global__ void AdditionKernel(const float* A, const float* B, float* C, unsigned int maxNum)
{
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < maxNum)
		C[i] = A[i] + B[i];
}

__global__ void TensorMatAdditionKernel(const float* tensor, const float* matrix, float* result, unsigned int tensorSize, unsigned int matrixSize)
{
    const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < tensorSize)
        result[i] = tensor[i] + matrix[i % matrixSize];
}

__global__ void SubtractionKernel(float* A, const float* B, unsigned int maxNum)
{
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < maxNum)
		A[i] -= B[i];
}

__global__ void TensorMatSubtractionKernel(float* tensor, const float* matrix, unsigned int tensorSize, unsigned int matrixSize)
{
    const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < tensorSize)
        tensor[i] -= matrix[i % matrixSize];
}

__global__ void SubtractionKernel(const float* A, const float* B, float* C, unsigned int maxNum)
{
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < maxNum)
		C[i] = A[i] - B[i];
}

__global__ void TensorMatSubtractionKernel(const float* tensor, const float* matrix, float* result, unsigned int tensorSize, unsigned int matrixSize)
{
    const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < tensorSize)
        result[i] = tensor[i] - matrix[i % matrixSize];
}

__global__ void InnerProductKernel(float* A, const float* B, unsigned int maxNum)
{
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < maxNum)
		A[i] *= B[i];
}

__global__ void FillKernel(float* a, float val, unsigned int maxNum)
{
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < maxNum)
		a[i] = val;
}

__global__ void ConstantAddingKernel(float* A, float v, unsigned int maxNum)
{
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < maxNum)
		A[i] += v;
}

__global__ void ConstantSubtractingKernel(float* A, float v, unsigned int maxNum)
{
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < maxNum)
		A[i] -= v;
}

__global__ void ConstantMultiplyingKernel(float* A, float v, unsigned int maxNum)
{
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < maxNum)
		A[i] *= v;
}

__global__ void CopyKernel(const float* from, float* to, unsigned int fromOffset, unsigned int toOffset, unsigned int count)
{
    const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < count)
        to[i + toOffset] = from[i + fromOffset];
}

//// Matrix math

// Addition
void MatrixCUDAMath::Add(const Matrix& a, const Matrix& b, Matrix& c)
{
	unsigned int max = a.GetColumnCount() * a.GetRowCount();
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    AdditionKernel <<<grid, threads >>>(a.GetConstGPUValues(), b.GetConstGPUValues(), c.GetGPUValues(), max);
}


void MatrixCUDAMath::AddIn(Matrix& a, const Matrix& b)
{
	unsigned int max = a.GetColumnCount() * a.GetRowCount();
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    AdditionKernel <<<grid, threads >>>(a.GetGPUValues(), b.GetConstGPUValues(), max);
}

void MatrixCUDAMath::AddConstant(Matrix& a, float v)
{
	unsigned int max = a.GetColumnCount() * a.GetRowCount();
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    ConstantAddingKernel <<<grid, threads >>>(a.GetGPUValues(), v, max);
}

// Subtraction
void MatrixCUDAMath::Subtract(const Matrix& a, const Matrix& b, Matrix& c)
{
	unsigned int max = a.GetColumnCount() * a.GetRowCount();
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    SubtractionKernel <<<grid, threads >>>(a.GetConstGPUValues(), b.GetConstGPUValues(), c.GetGPUValues(), max);
}

void MatrixCUDAMath::SubtractIn(Matrix& a, const Matrix& b)
{
	unsigned int max = a.GetColumnCount() * a.GetRowCount();
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    SubtractionKernel <<<grid, threads >>>(a.GetGPUValues(), b.GetConstGPUValues(), max);
}

void MatrixCUDAMath::SubtractConstant(Matrix& a, float v)
{
	unsigned int max = a.GetColumnCount() * a.GetRowCount();
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    ConstantSubtractingKernel <<<grid, threads >>>(a.GetGPUValues(), v, max);
}

//Multiplication
void MatrixCUDAMath::Multiplication(const Matrix& a, const Matrix& b, Matrix& c)
{
	unsigned int rows = ceil((double)(a.GetRowCount() + BLOCK_SIZE - 1) / (double)BLOCK_SIZE);
	unsigned int cols = ceil((double)(b.GetColumnCount() + BLOCK_SIZE - 1) / (double)BLOCK_SIZE);

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(cols, rows);
    MultiplicationKernel <<<blocks, threads >>>(a.GetConstGPUValues(), b.GetConstGPUValues(), c.GetGPUValues(),
                                                a.GetRowCount(), a.GetColumnCount(), b.GetColumnCount());
}

void MatrixCUDAMath::ElementwiseMultiply(Matrix& a, const Matrix& b)
{
	unsigned int max = a.GetColumnCount() * a.GetRowCount();
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
	InnerProductKernel <<<grid, threads >>> (a.GetGPUValues(), b.GetConstGPUValues(), max);
}

void MatrixCUDAMath::MultiplyConstant(Matrix& a, float v)
{
	unsigned int max = a.GetColumnCount() * a.GetRowCount();
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    ConstantMultiplyingKernel <<<grid, threads >>>(a.GetGPUValues(), v, max);
}

// Misc

void MatrixCUDAMath::FillWith(Matrix& a, float value)
{
	unsigned int max = a.GetColumnCount() * a.GetRowCount();
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
	FillKernel <<<grid, threads >>> (a.GetGPUValues(),value, max);
}

//// Tensor math

// Addition
void TensorCUDAMath::Add(const Tensor &a, const Tensor &b, Tensor &c)
{
    unsigned int max = a.GetElementCount();
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    AdditionKernel<<<grid, threads>>>(
            a.GetConstGPUValues(),
            b.GetConstGPUValues(),
            c.GetGPUValues(),
            max);
}

void TensorCUDAMath::AddIn(Tensor &a, const Tensor &b)
{
    unsigned int max = a.GetElementCount();
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    AdditionKernel<<<grid, threads>>>(
            a.GetGPUValues(),
            b.GetConstGPUValues(),
            max);
}

void TensorCUDAMath::AddIn(Tensor& a, const Matrix& b)
{
    unsigned int max = a.GetElementCount();
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    TensorMatAdditionKernel<<<grid, threads>>>(
            a.GetGPUValues(),
            b.GetConstGPUValues(),
            a.GetElementCount(),
            b.GetElementCount()
            );
}

void TensorCUDAMath::AddConstant(Tensor &a, float v)
{
    unsigned int max = a.GetShapeAt(0) * a.GetShapeAt(1);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    ConstantAddingKernel<<<grid, threads>>>(
            a.GetGPUValues(),
            v,
            max);

}

// Subtraction
void TensorCUDAMath::Subtract(const Tensor &a, const Tensor &b, Tensor &c)
{
    unsigned int max = a.GetElementCount();
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    SubtractionKernel<<<grid, threads>>>(
            a.GetConstGPUValues(),
            b.GetConstGPUValues(),
            c.GetGPUValues(),
            max);
}

void TensorCUDAMath::SubtractIn(Tensor &a, const Tensor &b)
{
    unsigned int max = a.GetElementCount();
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    SubtractionKernel<<<grid, threads>>>(
            a.GetGPUValues(),
            b.GetConstGPUValues(),
            max);
}

void TensorCUDAMath::SubtractIn(Tensor& a, const Matrix& b)
{
    unsigned int max = a.GetElementCount();
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    TensorMatSubtractionKernel<<<grid, threads>>>(
            a.GetGPUValues(),
            b.GetConstGPUValues(),
            a.GetElementCount(),
            b.GetElementCount()
    );
}

void TensorCUDAMath::SubtractConstant(Tensor &a, float v)
{
    unsigned int max = a.GetElementCount();
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    ConstantSubtractingKernel <<<grid, threads >>>(a.GetConstGPUValues(), v, max);
}

// Multiplication
void TensorCUDAMath::Multiplication(const Tensor &a, const Tensor &b, Tensor &c)
{
    unsigned int rows = ceil((double)(a.GetShapeAt(0) + BLOCK_SIZE - 1) / (double)BLOCK_SIZE);
    unsigned int cols = ceil((double)(b.GetShapeAt(1) + BLOCK_SIZE - 1) / (double)BLOCK_SIZE);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(cols, rows);
    for (unsigned int m = 0; m < a.GetMatrixCount(); m++)
    {
        MultiplicationKernel<<<blocks, threads>>>(
                a.GetConstGPUValues() + m * a.GetShapeAt(0) * a.GetShapeAt(1),
                b.GetConstGPUValues() + m * b.GetShapeAt(0) * b.GetShapeAt(1),
                c.GetGPUValues() + m * c.GetShapeAt(0) * c.GetShapeAt(1),
                a.GetShapeAt(0),
                a.GetShapeAt(1),
                b.GetShapeAt(1)
        );
    }
}

void TensorCUDAMath::Multiplication(const Tensor& a, const Matrix& b, Tensor& c)
{
    unsigned int rows = ceil((double)(a.GetShapeAt(0) + BLOCK_SIZE - 1) / (double)BLOCK_SIZE);
    unsigned int cols = ceil((double)(b.GetColumnCount() + BLOCK_SIZE - 1) / (double)BLOCK_SIZE);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(cols, rows);

    TensorMatMultiplicationKernel<<<blocks, threads>>>(
            a.GetConstGPUValues(),
            b.GetConstGPUValues(),
            c.GetGPUValues(),
            a.GetShapeAt(0),
            a.GetShapeAt(1),
            b.GetColumnCount(),
            a.GetMatrixCount()
            );
}

void TensorCUDAMath::MultiplyConstant(Tensor &a, float v)
{
    unsigned int max = a.GetElementCount();
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    ConstantMultiplyingKernel<<<grid, threads>>>(
            a.GetGPUValues(),
            v,
            max);
}

void TensorCUDAMath::ElementwiseMultiply(Tensor &a, const Tensor &b)
{
    unsigned int max = a.GetElementCount();
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    InnerProductKernel <<<grid, threads >>> (a.GetGPUValues(), b.GetConstGPUValues(), max);
}

// Misc

void TensorCUDAMath::FillWith(Tensor &a, float value)
{
    unsigned int max = a.GetElementCount();
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    FillKernel <<<grid, threads >>> (a.GetGPUValues(), value, max);
}

void CUDAOperations::CopyPartTo(Matrix& target, const Matrix& origin, unsigned int targetOffset,
                                unsigned int originOffset, unsigned int count)
{
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)count / (double)threads.x), ceil((double)count / (double)threads.y));
    CopyKernel<<<grid, threads>>>(origin.GetConstGPUValues(), target.GetGPUValues(), originOffset, targetOffset, count);
}

void CUDAOperations::CopyPartTo(Tensor& target, const Tensor& origin, unsigned int targetOffset,
                                unsigned int originOffset, unsigned int count)
{
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)count / (double)threads.x), ceil((double)count / (double)threads.y));
    CopyKernel<<<grid, threads>>>(origin.GetConstGPUValues(), target.GetGPUValues(), originOffset, targetOffset, count);
}

void CUDAOperations::CopyPartTo(Tensor& target, const Matrix& origin, unsigned int targetOffset,
                                unsigned int originOffset, unsigned int count)
{
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)count / (double)threads.x), ceil((double)count / (double)threads.y));
    CopyKernel<<<grid, threads>>>(origin.GetConstGPUValues(), target.GetGPUValues(), originOffset, targetOffset, count);
}