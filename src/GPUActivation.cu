#include "NeuralNetwork/GPUActivation.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "NeuralNetwork/Constants.h"

#define GLOBAL_ID (blockIdx.x + gridDim.x * blockIdx.y) * (blockDim.x * blockDim.y) + (threadIdx.x + blockDim.x * threadIdx.y)

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

__global__ void CUDASoftmax(float* from, float* to, unsigned int num)
{
    __shared__ float SUM;
    unsigned int usedMaximum = num;
    int id = GLOBAL_ID;
    if (id < num)
        to[id] = expf(from[id]);
    if (num % 2 == 1)
    {
        __threadfence();
        to[num - 2] += to[num - 1];
        usedMaximum--;
    }

    __syncthreads();
    for (unsigned int s = 1; s < usedMaximum; s *= 2)
    {
        if (id % (2 * s) == 0 && id < usedMaximum)
            to[id] += to[id + s];
        __syncthreads();
    }
    if (id == 0)
        SUM = to[0];
    __syncthreads();
    if (id < num)
        to[id] = expf(from[id]) / SUM;
}

__global__ void CUDASoftmaxInv(float* from, float* to, unsigned int num)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num)
        to[i] = from[i] * (1 - from[i]);
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

Matrix GPUActivation::SoftmaxCalculate(const Matrix& original)
{
    Matrix result(original);
    SoftmaxCalculate(original, result);
    return result;
}

void GPUActivation::SoftmaxCalculate(const Matrix& from, Matrix& to)
{
    unsigned int max = from.GetColumnCount() * from.GetRowCount();

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    CUDASoftmax<<<grid, threads >>> (from.GetConstGPUValues(), to.GetGPUValues(), from.GetElementCount());
}

Matrix GPUActivation::SoftmaxInvCalculate(const Matrix& original)
{
    Matrix result(original);
    SoftmaxInvCalculate(original, result);
    return result;
}

void GPUActivation::SoftmaxInvCalculate(const Matrix& from, Matrix& to)
{
    unsigned int max = from.GetColumnCount() * from.GetRowCount();

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    CUDASigmoidInv<<<grid, threads >>> (from.GetConstGPUValues(), to.GetGPUValues(), from.GetElementCount());
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

Tensor GPUActivation::SoftmaxCalculate(const Tensor& original)
{
    Tensor result(original);
    SoftmaxCalculate(original, result);
    return result;
}

void GPUActivation::SoftmaxCalculate(const Tensor& from, Tensor& to)
{
    unsigned int max = from.GetElementCount();

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    CUDASoftmax<<<grid, threads >>> (from.GetConstGPUValues(), to.GetGPUValues(), from.GetElementCount());
}

Tensor GPUActivation::SoftmaxInvCalculate(const Tensor& original)
{
    Tensor result(original);
    SoftmaxInvCalculate(original, result);
    return result;
}

void GPUActivation::SoftmaxInvCalculate(const Tensor& from, Tensor& to)
{
    unsigned int max = from.GetElementCount();

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double)max / (double)threads.x), ceil((double)max / (double)threads.y));
    CUDASoftmaxInv<<<grid, threads >>> (from.GetConstGPUValues(), to.GetGPUValues(), from.GetElementCount());
}