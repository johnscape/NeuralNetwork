#include <cuda_runtime.h>
#include "NeuralNetwork/CUDAMath.cuh"
#if USE_CUBLAS
#include <cublas_v2.h>
#endif

// Kernels

// A: aRows x aCols -
// B: aCols x bCols - x m
// C: aRows x bCols -
__global__ void MatMulKernel(const float* A, const float* B, float* C, int aRows, int aCols, int bCols)
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

__global__ void MatAddInKernel(float* A, const float* B, unsigned int maxNum)
{
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < maxNum)
		A[i] += B[i];
}

__global__ void MatAddKernel(const float* A, const float* B, float* C, unsigned int maxNum)
{
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < maxNum)
		C[i] = A[i] + B[i];
}

__global__ void MatSubInKernel(float* A, const float* B, unsigned int maxNum)
{
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < maxNum)
		A[i] -= B[i];
}

__global__ void MatSubKernel(const float* A, const float* B, float* C, unsigned int maxNum)
{
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < maxNum)
		C[i] = A[i] - B[i];
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

__global__ void AddConstKernel(float* A, float v, unsigned int maxNum)
{
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < maxNum)
		A[i] += v;
}

__global__ void SubConstKernel(float* A, float v, unsigned int maxNum)
{
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < maxNum)
		A[i] -= v;
}

__global__ void MulConstKernel(float* A, float v, unsigned int maxNum)
{
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < maxNum)
		A[i] *= v;
}

//// Matrix math

// Addition
void MatrixCUDAMath::Add(const Matrix& a, const Matrix& b, Matrix& c)
{
	unsigned int max = a.GetColumnCount() * a.GetRowCount();
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    MatAddKernel <<<grid, threads >>>(a.GetConstGPUValues(), b.GetConstGPUValues(), c.GetGPUValues(), max);
}


void MatrixCUDAMath::AddIn(Matrix& a, const Matrix& b)
{
	unsigned int max = a.GetColumnCount() * a.GetRowCount();
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
	MatAddInKernel <<<grid, threads >>> (a.GetGPUValues(), b.GetConstGPUValues(), max);
}

void MatrixCUDAMath::AddConstant(Matrix& a, float v)
{
	unsigned int max = a.GetColumnCount() * a.GetRowCount();
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
	AddConstKernel <<<grid, threads >>> (a.GetGPUValues(), v, max);
}

// Subtraction
void MatrixCUDAMath::Subtract(const Matrix& a, const Matrix& b, Matrix& c)
{
	unsigned int max = a.GetColumnCount() * a.GetRowCount();
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
	MatSubKernel <<<grid, threads >>> (a.GetConstGPUValues(), b.GetConstGPUValues(), c.GetGPUValues(), max);
}

void MatrixCUDAMath::SubtractIn(Matrix& a, const Matrix& b)
{
	unsigned int max = a.GetColumnCount() * a.GetRowCount();
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
	MatSubInKernel <<<grid, threads >>> (a.GetGPUValues(), b.GetConstGPUValues(), max);
}

void MatrixCUDAMath::SubtractConstant(Matrix& a, float v)
{
	unsigned int max = a.GetColumnCount() * a.GetRowCount();
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
	SubConstKernel <<<grid, threads >>> (a.GetGPUValues(), v, max);
}

//Multiplication
void MatrixCUDAMath::Multiplication(const Matrix& a, const Matrix& b, Matrix& c)
{
	unsigned int rows = ceil((double)(a.GetRowCount() + BLOCK_SIZE - 1) / (double)BLOCK_SIZE);
	unsigned int cols = ceil((double)(b.GetColumnCount() + BLOCK_SIZE - 1) / (double)BLOCK_SIZE);

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(cols, rows);
	MatMulKernel <<<blocks, threads >>> (a.GetConstGPUValues(), b.GetConstGPUValues(), c.GetGPUValues(), a.GetRowCount(), a.GetColumnCount(), b.GetColumnCount());
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
	MulConstKernel <<<grid, threads >>> (a.GetGPUValues(), v, max);
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
void TensorMath::Add(const Tensor &a, const Tensor &b, Tensor &c)
{
    unsigned int max = a.GetElementCount();
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    MatAddKernel<<<grid, threads>>>(
            a.GetConstGPUValues(),
            b.GetConstGPUValues(),
            c.GetGPUValues(),
            max);
}

void TensorMath::AddIn(Tensor &a, const Tensor &b)
{
    unsigned int max = a.GetShapeAt(0) * a.GetShapeAt(1);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    MatAddInKernel<<<grid, threads>>>(
            a.GetGPUValues(),
            b.GetConstGPUValues(),
            max);
}

void TensorMath::AddIn(Tensor& a, const Matrix& b)
{
    unsigned int max = a.GetShapeAt(0) * a.GetShapeAt(1);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    MatAddInKernel<<<grid, threads>>>(
            a.GetGPUValues(),
            b.GetConstGPUValues(),
            max);
}

void TensorMath::AddConstant(Tensor &a, float v)
{
    unsigned int max = a.GetShapeAt(0) * a.GetShapeAt(1);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    AddConstKernel<<<grid, threads>>>(
            a.GetGPUValues(),
            v,
            max);

}

// Subtraction
void TensorMath::Subtract(const Tensor &a, const Tensor &b, Tensor &c)
{
    unsigned int max = a.GetElementCount();
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    MatSubKernel<<<grid, threads>>>(
            a.GetConstGPUValues(),
            b.GetConstGPUValues(),
            c.GetGPUValues(),
            max);
}

void TensorMath::SubtractIn(Tensor &a, const Tensor &b)
{
    unsigned int max = a.GetElementCount();
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    MatSubInKernel<<<grid, threads>>>(
            a.GetGPUValues(),
            b.GetConstGPUValues(),
            max);
}

void TensorMath::SubtractIn(Tensor& a, const Matrix& b)
{
    unsigned int max = a.GetElementCount();
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    MatSubInKernel<<<grid, threads>>>(
            a.GetGPUValues(),
            b.GetConstGPUValues(),
            max);
}

void TensorMath::SubtractConstant(Tensor &a, float v)
{
    unsigned int max = a.GetElementCount();
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    SubConstKernel <<<grid, threads >>> (a.GetConstGPUValues(), v, max);
}

// Multiplication
void TensorMath::Multiplication(const Tensor &a, const Tensor &b, Tensor &c)
{
    unsigned int rows = ceil((double)(a.GetShapeAt(0) + BLOCK_SIZE - 1) / (double)BLOCK_SIZE);
    unsigned int cols = ceil((double)(b.GetShapeAt(1) + BLOCK_SIZE - 1) / (double)BLOCK_SIZE);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(cols, rows);
    for (unsigned int m = 0; m < a.GetMatrixCount(); m++)
    {
        MatMulKernel<<<blocks, threads>>>(
                a.GetConstGPUValues() + m * rows * cols,
                b.GetConstGPUValues() + m * rows * cols,
                c.GetGPUValues() + m * rows * cols,
                a.GetShapeAt(0),
                a.GetShapeAt(1),
                b.GetShapeAt(1)
                );
    }
}

void TensorMath::Multiplication(const Tensor& a, const Matrix& b, Tensor& c)
{
    unsigned int rows = ceil((double)(a.GetShapeAt(0) + BLOCK_SIZE - 1) / (double)BLOCK_SIZE);
    unsigned int cols = ceil((double)(b.GetColumnCount() + BLOCK_SIZE - 1) / (double)BLOCK_SIZE);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(cols, rows);
    for (unsigned int m = 0; m < a.GetMatrixCount(); m++)
    {
        MatMulKernel<<<blocks, threads>>>(
                a.GetConstGPUValues() + m * rows * cols,
                b.GetConstGPUValues(),
                c.GetGPUValues() + m * rows * cols,
                a.GetShapeAt(0),
                a.GetShapeAt(1),
                b.GetColumnCount()
        );
    }
}

void TensorMath::MultiplyConstant(Tensor &a, float v)
{
    unsigned int max = a.GetElementCount();
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    MulConstKernel<<<grid, threads>>>(
            a.GetGPUValues(),
            v,
            max);
}

void TensorMath::ElementwiseMultiply(Tensor &a, const Tensor &b)
{
    unsigned int max = a.GetElementCount();
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    InnerProductKernel <<<grid, threads >>> (a.GetGPUValues(), b.GetConstGPUValues(), max);
}

// Misc

void TensorMath::FillWith(Tensor &a, float value)
{
    unsigned int max = a.GetElementCount();
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    FillKernel <<<grid, threads >>> (a.GetGPUValues(), value, max);
}