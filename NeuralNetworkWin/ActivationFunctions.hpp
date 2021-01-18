#pragma once

#include "Matrix.h"
#include "MatrixMath.h"
#include <math.h>
#include <memory>
#include "Constants.h"

#if USE_GPU
#include "GPUActivation.cuh"
#endif // USE_GPU


//TODO: Create singletons

/**
 * @brief Abstract class for handling activation functions
*/
class ActivationFunction
{
public:
	ActivationFunction() {}
	~ActivationFunction() {}

	virtual Matrix* CalculateMatrix(Matrix* input) = 0;
	virtual Matrix* CalculateDerivateMatrix(Matrix* output, float extra = 0) = 0;

	virtual void CalculateInto(Matrix* input, Matrix* target) = 0;
	virtual void CalculateDerivateInto(Matrix* output, Matrix* target, float extra = 0) = 0;
};

/**
 * @brief Indentity activation function
*/
class IdentityFunction : public ActivationFunction
{
public:
	static IdentityFunction& GetInstance()
	{
		static IdentityFunction func;
		return func;
	}

	IdentityFunction(const IdentityFunction& f) = delete;
	void operator=(const IdentityFunction& f) = delete;

	~IdentityFunction() {}
	virtual Matrix* CalculateMatrix(Matrix* input) { Matrix* c = new Matrix(*input); return c; }
	virtual Matrix* CalculateDerivateMatrix(Matrix* output, float extra = 0) { Matrix* c = new Matrix(*output); MatrixMath::FillWith(c, 1); return c; }

	virtual void CalculateInto(Matrix* input, Matrix* target) { MatrixMath::Copy(input, target); }
	virtual void CalculateDerivateInto(Matrix* output, Matrix* target, float extra = 0) { MatrixMath::FillWith(target, 1); }
private:
	IdentityFunction() {}

};

//TODO: Add Binary step

/**
 * @brief Sigmoid activation function
*/
class Sigmoid : public ActivationFunction
{
public:
	static Sigmoid& GetInstance()
	{
		static Sigmoid func;
		return func;
	}

	Sigmoid(const Sigmoid& f) = delete;
	void operator=(const Sigmoid& f) = delete;

	~Sigmoid() {}
	virtual Matrix* CalculateMatrix(Matrix* input)
	{
#if USE_GPU
		return GPUActivation::SigmoidCalculate(input);
#else
		Matrix* c = new Matrix(*input);
		for (size_t i = 0; i < c->GetRowCount() * c->GetColumnCount(); i++)
			c->SetValue(i, Calculate(c->GetValue(i)));
		return c;
#endif // USE_GPU
	}
	virtual Matrix* CalculateDerivateMatrix(Matrix* output, float extra = 0)
	{
#if USE_GPU
		return GPUActivation::SigmoidInvCalculate(output);
#else
		Matrix* c = new Matrix(*output);
		for (size_t i = 0; i < c->GetRowCount() * c->GetColumnCount(); i++)
			c->SetValue(i, CalculateDerivate(c->GetValue(i)));
		return c;
#endif // USE_GPU

	}

	virtual void CalculateInto(Matrix* input, Matrix* target)
	{
#if USE_GPU
		GPUActivation::SigmoidCalculate(input, target);
#else
		for (size_t i = 0; i < input->GetRowCount() * input->GetColumnCount(); i++)
			target->SetValue(i, Calculate(input->GetValue(i)));
#endif // USE_GPU

	}

	virtual void CalculateDerivateInto(Matrix* output, Matrix* target, float extra = 0)
	{
#if USE_GPU
		GPUActivation::SigmoidInvCalculate(output, target);
#else
		for (size_t i = 0; i < output->GetRowCount() * output->GetColumnCount(); i++)
			target->SetValue(i, CalculateDerivate(output->GetValue(i)));
#endif // USE_GPU

	}

private:
	float Calculate(float a) { return  1.0f / (1.0f + exp(-a)); }
	float CalculateDerivate(float a) { return a * (1 - a); }
	Sigmoid() {}
};

/**
 * @brief Tanh activation function
*/
class TanhFunction : public ActivationFunction
{
public:
	static TanhFunction& GetInstance()
	{
		static TanhFunction func;
		return func;
	}

	TanhFunction(const TanhFunction& f)		 = delete;
	void operator=(const TanhFunction& f)	 = delete;

	~TanhFunction() {}
	virtual Matrix* CalculateMatrix(Matrix* input)
	{
#if USE_GPU
		return GPUActivation::TanhCalculate(input);
#else
		Matrix* c = new Matrix(*input);
		for (size_t i = 0; i < c->GetRowCount() * c->GetColumnCount(); i++)
			c->SetValue(i, Calculate(c->GetValue(i)));
		return c;
#endif // USE_GPU

	}
	virtual Matrix* CalculateDerivateMatrix(Matrix* output, float extra = 0)
	{
#if USE_GPU
		return GPUActivation::TanhInvCalculate(output);
#else
		Matrix* c = new Matrix(*output);
		for (size_t i = 0; i < c->GetRowCount() * c->GetColumnCount(); i++)
			c->SetValue(i, CalculateDerivate(c->GetValue(i)));
		return c;
#endif // USE_GPU

	}

	virtual void CalculateInto(Matrix* input, Matrix* target)
	{
#if USE_GPU
		GPUActivation::TanhCalculate(input, target);
#else
		for (size_t i = 0; i < input->GetRowCount() * input->GetColumnCount(); i++)
			target->SetValue(i, Calculate(input->GetValue(i)));
#endif // USE_GPU

	}

	virtual void CalculateDerivateInto(Matrix* output, Matrix* target, float extra = 0)
	{
#if USE_GPU
		GPUActivation::TanhInvCalculate(output, target);
#else
		for (size_t i = 0; i < output->GetRowCount() * output->GetColumnCount(); i++)
			target->SetValue(i, CalculateDerivate(output->GetValue(i)));
#endif // USE_GPU

	}

private:
	float Calculate(float a) {
		return tanh(a);
	}

	float CalculateDerivate(float a) { return (float)(1 - pow(a, 2)); }

	TanhFunction() {}
};
