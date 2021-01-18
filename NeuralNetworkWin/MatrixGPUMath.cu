#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "MatrixGPUMath.cuh"

//code is from NVidia MatMul example project
template <int BLOCK_SIZE> __global__ void MatMulKernel(float* A, float* B, float* C, int wA, int wB)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int aBegin = wA * BLOCK_SIZE * by;
	int aEnd = aBegin + wA - 1;
	int aStep = BLOCK_SIZE;

	int bBegin = BLOCK_SIZE * bx;
	int bStep = BLOCK_SIZE * wB;

	float Csub = 0;

	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
	{
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		As[ty][tx] = A[a + wA * ty + tx];
		Bs[ty][tx] = B[b + wB * ty + tx];

		__syncthreads();

		for (int k = 0; k < BLOCK_SIZE; ++k)
			Csub += As[ty][k] * Bs[k][tx];

		__syncthreads();
	}

	int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	C[c + wB * ty + tx] = Csub;
}

__global__ void MatAddInKernel(float* A, float* B, unsigned int maxNum)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < maxNum)
		A[i] += B[i];
}


unsigned int GPUMath::CalculateMaxBlockSize(Matrix* a, Matrix* b, unsigned int max)
{
	unsigned int blockSize = 1;
	unsigned int smallest = a->GetColumnCount();
	if (a->GetRowCount() < smallest)
		smallest = a->GetRowCount();
	if (b)
	{
		if (b->GetColumnCount() < smallest)
			smallest = b->GetColumnCount();
		if (b->GetRowCount() < smallest)
			smallest = b->GetRowCount();
	}

	while (blockSize < max)
	{
		blockSize *= 2;
		if (blockSize > smallest)
		{
			blockSize /= 2;
			break;
		}
	}

	return blockSize;
}

Matrix* GPUMath::Multiplication(Matrix* a, Matrix* b, Matrix* c)
{
	if (!c)
		c = new Matrix(a->GetRowCount(), b->GetColumnCount());
	unsigned int blockSize = CalculateMaxBlockSize(a, b, 16);
	dim3 threads(blockSize, blockSize);
	dim3 grid(b->GetColumnCount() / threads.x, a->GetRowCount() / threads.y);
	MatMulKernel<16> <<<grid, threads>>> (a->GetGPUValues(), b->GetGPUValues(), c->GetGPUValues(), a->GetColumnCount(), b->GetColumnCount());
	return c;
}

void GPUMath::AddIn(Matrix* a, Matrix* b)
{
	unsigned int blockSize = CalculateMaxBlockSize(a, b, 16);
	unsigned int max = a->GetColumnCount() * a->GetRowCount();
	dim3 threads(blockSize, blockSize);
	dim3 grid(b->GetColumnCount() / threads.x, a->GetRowCount() / threads.y);
	MatAddInKernel <<<grid, threads >>> (a->GetGPUValues(), b->GetGPUValues(), max);
}
