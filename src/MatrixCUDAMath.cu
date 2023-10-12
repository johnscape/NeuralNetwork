#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "NeuralNetwork/MatrixCUDAMath.cuh"

// Kernels

__global__ void MatMulKernel(float* A, float* B, float* C, int m, int n, int k)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0;
	if (col < k && row < m)
	{
		for (int i = 0; i < n; i++)
		{
			sum += A[row * n + i] * B[i * k + col];
		}
		C[row * k + col] = sum;
	}
}

__global__ void MatAddInKernel(float* A, float* B, unsigned int maxNum)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < maxNum)
		A[i] += B[i];
}

__global__ void MatAddKerlen(float* A, float* B, float* C, unsigned int maxNum)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < maxNum)
		C[i] = A[i] + B[i];
}

__global__ void MatSubInKernel(float* A, float* B, unsigned int maxNum)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < maxNum)
		A[i] -= B[i];
}

__global__ void MatSubKernel(float* A, float* B, float* C, unsigned int maxNum)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < maxNum)
		C[i] = A[i] - B[i];
}

__global__ void InnerProductKernel(float* A, float* B, unsigned int maxNum)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < maxNum)
		A[i] *= B[i];
}

__global__ void FillKernel(float* a, float val, unsigned int maxNum)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < maxNum)
		a[i] = val;
}

__global__ void AddConstKernel(float* A, float v, unsigned int maxNum)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < maxNum)
		A[i] += v;
}

__global__ void SubConstKernel(float* A, float v, unsigned int maxNum)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < maxNum)
		A[i] -= v;
}

__global__ void MulConstKernel(float* A, float v, unsigned int maxNum)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < maxNum)
		A[i] *= v;
}

// Addition
void MatrixCUDAMath::Add(const Matrix& a, const Matrix& b, Matrix& c)
{
	unsigned int max = a.GetColumnCount() * a.GetRowCount();
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
	MatAddKerlen <<<grid, threads >>> (a.GetConstGPUValues(), b.GetConstGPUValues(), c.GetGPUValues(), max);
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
	//cudaMemset(a.GetGPUValues(), value, a.GetRowCount() * a.GetColumnCount() * sizeof(float));
	unsigned int blockSize = 1;//CalculateMaxBlockSize(a, nullptr, 16);
	unsigned int max = a.GetColumnCount() * a.GetRowCount();
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
	FillKernel <<<grid, threads >>> (a.GetGPUValues(),value, max);
}
