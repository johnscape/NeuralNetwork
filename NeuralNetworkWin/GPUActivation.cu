#include "GPUActivation.cuh"
#include "MatrixGPUMath.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "MatrixMath.h"

__global__ void CUDASigmoid(float* from, float* to, unsigned int num)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < num)
		to[i] = 1.0 / (1.0 + exp(-from[i]));
}

__global__ void CUDATanh(float* from, float* to, unsigned int num)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < num)
		to[i] = tanh(from[i]);
}

__global__ void CUDASigmoidInv(float* from, float* to, unsigned int num)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < num)
		to[i] = from[i] * (1 - from[i]);
}

__global__ void CUDATanhInv(float* from, float* to, unsigned int num)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < num)
		to[i] = 1.0 - powf(from[i], 2);
}

Matrix* GPUActivation::SigmoidCalculate(Matrix* original)
{
	unsigned int max = original->GetColumnCount() * original->GetRowCount();
	Matrix* ret = new Matrix(original->GetRowCount(), original->GetColumnCount());
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
	CUDASigmoid <<<grid, threads>>> (original->GetGPUValues(), ret->GetGPUValues(), original->GetRowCount() * original->GetColumnCount());
	return ret;
}

void GPUActivation::SigmoidCalculate(Matrix* from, Matrix* to)
{
	unsigned int max = from->GetColumnCount() * from->GetRowCount();
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
	CUDASigmoid << <grid, threads >> > (from->GetGPUValues(), to->GetGPUValues(), from->GetRowCount() * from->GetColumnCount());
}

Matrix* GPUActivation::SigmoidInvCalculate(Matrix* original)
{
	unsigned int max = original->GetColumnCount() * original->GetRowCount();
	Matrix* ret = new Matrix(original->GetRowCount(), original->GetColumnCount());
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
	CUDASigmoidInv << <grid, threads >> > (original->GetGPUValues(), ret->GetGPUValues(), original->GetRowCount() * original->GetColumnCount());
	return ret;
}

void GPUActivation::SigmoidInvCalculate(Matrix* from, Matrix* to)
{
	unsigned int max = from->GetColumnCount() * from->GetRowCount();
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
	CUDASigmoidInv << <grid, threads >> > (from->GetGPUValues(), to->GetGPUValues(), from->GetRowCount() * from->GetColumnCount());
}

Matrix* GPUActivation::TanhCalculate(Matrix* original)
{
	unsigned int max = original->GetColumnCount() * original->GetRowCount();
	Matrix* ret = new Matrix(original->GetRowCount(), original->GetColumnCount());
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
	CUDATanh << <grid, threads >> > (original->GetGPUValues(), ret->GetGPUValues(), original->GetRowCount() * original->GetColumnCount());
	return ret;
}

void GPUActivation::TanhCalculate(Matrix* from, Matrix* to)
{
	unsigned int max = from->GetColumnCount() * from->GetRowCount();
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
	CUDATanh << <grid, threads >> > (from->GetGPUValues(), to->GetGPUValues(), from->GetRowCount() * from->GetColumnCount());
}

Matrix* GPUActivation::TanhInvCalculate(Matrix* original)
{
	unsigned int max = original->GetColumnCount() * original->GetRowCount();
	Matrix* ret = new Matrix(original->GetRowCount(), original->GetColumnCount());
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
	CUDATanhInv << <grid, threads >> > (original->GetGPUValues(), ret->GetGPUValues(), original->GetRowCount() * original->GetColumnCount());
	return ret;
}

void GPUActivation::TanhInvCalculate(Matrix* from, Matrix* to)
{
	unsigned int max = from->GetColumnCount() * from->GetRowCount();
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
	CUDATanhInv <<<grid, threads>>> (from->GetGPUValues(), to->GetGPUValues(), from->GetRowCount() * from->GetColumnCount());
}
