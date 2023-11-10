#include "NeuralNetwork/GPUActivation.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "NeuralNetwork/Constants.h"

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

// Matrices

Matrix GPUActivation::SigmoidCalculate(const Matrix& original)
{
    Matrix result(original);
    SigmoidCalculate(original, result);
    return result;
}

void GPUActivation::SigmoidCalculate(const Matrix& from, Matrix& to)
{
	unsigned int max = from.GetColumnCount() * from.GetRowCount();

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
	CUDASigmoid <<<grid, threads >>> (from.GetConstGPUValues(), to.GetGPUValues(), from.GetElementCount());

}

Matrix GPUActivation::SigmoidInvCalculate(const Matrix& original)
{
	Matrix result(original);
    SigmoidInvCalculate(original, result);
    return result;
}

void GPUActivation::SigmoidInvCalculate(const Matrix& from, Matrix& to)
{
	unsigned int max = from.GetColumnCount() * from.GetRowCount();

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
	CUDASigmoidInv <<<grid, threads >>> (from.GetConstGPUValues(), to.GetGPUValues(), from.GetElementCount());
	}

Matrix GPUActivation::TanhCalculate(const Matrix& original)
{
	Matrix result(original);
    TanhCalculate(original, result);
    return result;
}

void GPUActivation::TanhCalculate(const Matrix& from, Matrix& to)
{
	unsigned int max = from.GetColumnCount() * from.GetRowCount();

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
	CUDATanh <<<grid, threads >>> (from.GetConstGPUValues(), to.GetGPUValues(), from.GetElementCount());
}

Matrix GPUActivation::TanhInvCalculate(const Matrix& original)
{
	Matrix result(original);
    TanhInvCalculate(original, result);
    return result;
}

void GPUActivation::TanhInvCalculate(const Matrix& from, Matrix& to)
{
	unsigned int max = from.GetColumnCount() * from.GetRowCount();

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
	CUDATanhInv <<<grid, threads >>> (from.GetConstGPUValues(), to.GetGPUValues(), from.GetElementCount());
}

// Tensors

Tensor GPUActivation::SigmoidCalculate(const Tensor& original)
{
    Tensor result(original);
    SigmoidCalculate(original, result);
    return result;
}

void GPUActivation::SigmoidCalculate(const Tensor& from, Tensor& to)
{
    unsigned int max = from.GetElementCount();

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    CUDASigmoid <<<grid, threads >>> (from.GetConstGPUValues(), to.GetGPUValues(), from.GetElementCount());

}

Tensor GPUActivation::SigmoidInvCalculate(const Tensor& original)
{
    Tensor result(original);
    SigmoidInvCalculate(original, result);
    return result;
}

void GPUActivation::SigmoidInvCalculate(const Tensor& from, Tensor& to)
{
    unsigned int max = from.GetElementCount();

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    CUDASigmoidInv <<<grid, threads >>> (from.GetConstGPUValues(), to.GetGPUValues(), from.GetElementCount());
}

Tensor GPUActivation::TanhCalculate(const Tensor& original)
{
    Tensor result(original);
    TanhCalculate(original, result);
    return result;
}

void GPUActivation::TanhCalculate(const Tensor& from, Tensor& to)
{
    unsigned int max = from.GetElementCount();

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    CUDATanh <<<grid, threads >>> (from.GetConstGPUValues(), to.GetGPUValues(), from.GetElementCount());
}

Tensor GPUActivation::TanhInvCalculate(const Tensor& original)
{
    Tensor result(original);
    TanhInvCalculate(original, result);
    return result;
}

void GPUActivation::TanhInvCalculate(const Tensor& from, Tensor& to)
{
    unsigned int max = from.GetElementCount();

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    CUDATanhInv <<<grid, threads >>> (from.GetConstGPUValues(), to.GetGPUValues(), from.GetElementCount());
}