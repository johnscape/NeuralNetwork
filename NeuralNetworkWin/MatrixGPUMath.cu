#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "MatrixGPUMath.cuh"


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

__global__ void MatSubInKernel(float* A, float* B, unsigned int maxNum)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < maxNum)
		A[i] -= B[i];
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

Matrix* GPUMath::Multiplication(Matrix* a, Matrix* b, Matrix* c)
{
	if (!c)
		c = new Matrix(a->GetRowCount(), b->GetColumnCount());
	unsigned int rows = ceil((double)(a->GetRowCount() + BLOCK_SIZE - 1) / (double)BLOCK_SIZE);
	unsigned int cols = ceil((double)(b->GetColumnCount() + BLOCK_SIZE - 1) / (double)BLOCK_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(cols, rows);
	MatMulKernel<<<blocks, threads>>> (a->GetGPUValues(), b->GetGPUValues(), c->GetGPUValues(), a->GetRowCount(), a->GetColumnCount(), b->GetColumnCount());
	return c;
}

void GPUMath::ElementviseMultiply(Matrix* a, Matrix* b)
{
	unsigned int max = a->GetColumnCount() * a->GetRowCount();
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
	InnerProductKernel << <grid, threads >> > (a->GetGPUValues(), b->GetGPUValues(), max);
}

void GPUMath::SubstractIn(Matrix* a, Matrix* b)
{
	unsigned int max = a->GetColumnCount() * a->GetRowCount();
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
	MatSubInKernel << <grid, threads >> > (a->GetGPUValues(), b->GetGPUValues(), max);
}

void GPUMath::AddIn(Matrix* a, Matrix* b)
{
	unsigned int max = a->GetColumnCount() * a->GetRowCount();
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
	MatAddInKernel <<<grid, threads >>> (a->GetGPUValues(), b->GetGPUValues(), max);
}

void GPUMath::FillWith(Matrix* a, float value)
{
	//cudaMemset(a->GetGPUValues(), value, a->GetRowCount() * a->GetColumnCount() * sizeof(float));
	unsigned int blockSize = 1;//CalculateMaxBlockSize(a, nullptr, 16);
	unsigned int max = a->GetColumnCount() * a->GetRowCount();
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
	FillKernel <<<grid, threads >>> (a->GetGPUValues(),value, max);
}
