#pragma once

#include "Matrix.h"
#include <math.h>
#include <memory>
#include "Constants.h"

#define ACTIVATION_SIGMOID &Sigmoid::GetInstance()
#define ACTIVATION_TANH &TanhFunction::GetInstance()
#define ACTIVATION_LINEAR &IdentityFunction::GetInstance()

#if USE_GPU
#include "GPUActivation.cuh"
#endif // USE_GPU

/**
 * @brief Abstract class for handling activation functions
*/
class ActivationFunction
{
public:
	ActivationFunction() = default;
	~ActivationFunction() = default;

	virtual Matrix CalculateMatrix(const Matrix& input) = 0;
	virtual Matrix CalculateDerivateMatrix(const Matrix& output, float extra) = 0;
	Matrix CalculateDerivateMatrix(const Matrix& output) {return CalculateDerivateMatrix(output, 0);}

	virtual void CalculateInto(const Matrix& input, Matrix& target) = 0;
	virtual void CalculateDerivateInto(const Matrix& output, Matrix& target, float extra) = 0;
	void CalculateDerivateInto(const Matrix& output, Matrix& target) {CalculateDerivateInto(output, target, 0);}

	virtual Tensor CalculateTensor(const Tensor& input) = 0;
	virtual Tensor CalculateDerivateTensor(const Tensor& output, float extra) = 0;
	Tensor CalculateDerivateTensor(const Tensor& output) {return CalculateDerivateTensor(output, 0);}

	virtual void CalculateInto(const Tensor& input, Tensor& target) = 0;
	virtual void CalculateDerivateInto(const Tensor& output, Tensor& target, float extra) = 0;
	void CalculateDerivateInto(const Tensor& output, Tensor& target) {CalculateDerivateInto(output, target, 0);}
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

	~IdentityFunction() = default;
	Matrix CalculateMatrix(const Matrix& input) override
	{ 
		Matrix c(input);
		return c;
	}

	Matrix CalculateDerivateMatrix(const Matrix& output, float extra) override
	{
		Matrix c(output);
		c.FillWith(1);
		return c;
	}

	void CalculateInto(const Matrix& input, Matrix& target) override
	{ 
		target.Copy(input);
	}

	void CalculateDerivateInto(const Matrix& output, Matrix& target, float extra) override
	{ 
		target.FillWith(1);
	}

	Tensor CalculateTensor(const Tensor& input) override
	{
		Tensor t(input);
		return t;
	}

	Tensor CalculateDerivateTensor(const Tensor& output, float extra) override
	{
		Tensor t(output.GetShape());
		t.FillWith(1);
		return t;
	}

	void CalculateInto(const Tensor& input, Tensor& target) override
	{
		target.Copy(input);
	}

	void CalculateDerivateInto(const Tensor& output, Tensor& target, float extra) override
	{
		target.FillWith(1);
	}

private:
	IdentityFunction() = default;

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

	~Sigmoid() = default;
	Matrix CalculateMatrix(const Matrix& input) override
	{
#if USE_GPU
		return GPUActivation::SigmoidCalculate(input);
#else
		Matrix c(input);
		for (size_t i = 0; i < c.GetRowCount() * c.GetColumnCount(); i++)
			c.SetValue(i, Calculate(c.GetValue(i)));
		return c;
#endif // USE_GPU
	}
	Matrix CalculateDerivateMatrix(const Matrix& output, float extra) override
	{
#if USE_GPU
		return GPUActivation::SigmoidInvCalculate(output);
#else
		Matrix c(output);
		for (size_t i = 0; i < c.GetRowCount() * c.GetColumnCount(); i++)
			c.SetValue(i, CalculateDerivate(c.GetValue(i)));
		return c;
#endif // USE_GPU

	}

	void CalculateInto(const Matrix& input, Matrix& target) override
	{
#if USE_GPU
		GPUActivation::SigmoidCalculate(input, target);
#else
		for (size_t i = 0; i < input.GetRowCount() * input.GetColumnCount(); i++)
			target.SetValue(i, Calculate(input.GetValue(i)));
#endif // USE_GPU

	}

	void CalculateDerivateInto(const Matrix& output, Matrix& target, float extra) override
	{
#if USE_GPU
		GPUActivation::SigmoidInvCalculate(output, target);
#else
		for (size_t i = 0; i < output.GetRowCount() * output.GetColumnCount(); i++)
			target.SetValue(i, CalculateDerivate(output.GetValue(i)));
#endif // USE_GPU

	}

	Tensor CalculateTensor(const Tensor& input) override
	{
		Tensor t(input);

		for (unsigned int i = 0; i < t.GetElementCount(); ++i)
			t.SetValue(i, Calculate(t.GetValue(i)));

		return t;
	}

	Tensor CalculateDerivateTensor(const Tensor& output, float extra) override
	{
		Tensor t(output);

		for (unsigned int i = 0; i < t.GetElementCount(); ++i)
			t.SetValue(i, CalculateDerivate(t.GetValue(i)));

		return t;
	}

	void CalculateInto(const Tensor& input, Tensor& target) override
	{
		for (unsigned int i = 0; i < input.GetElementCount(); ++i)
			target.SetValue(i, Calculate(input.GetValue(i)));
	}

	void CalculateDerivateInto(const Tensor& output, Tensor& target, float extra) override
	{
		for (unsigned int i = 0; i < output.GetElementCount(); ++i)
			target.SetValue(i, CalculateDerivate(output.GetValue(i)));
	}

private:
	static float Calculate(float a) { return  1.0f / (1.0f + exp(-a)); }
	static float CalculateDerivate(float a) { return a * (1 - a); }
	Sigmoid() = default;
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

	~TanhFunction() = default;
	Matrix CalculateMatrix(const Matrix& input) override
	{
#if USE_GPU
		return GPUActivation::TanhCalculate(input);
#else
		Matrix c(input);
		for (size_t i = 0; i < c.GetRowCount() * c.GetColumnCount(); i++)
			c.SetValue(i, Calculate(c.GetValue(i)));
		return c;
#endif // USE_GPU

	}
	Matrix CalculateDerivateMatrix(const Matrix& output, float extra) override
	{
#if USE_GPU
		return GPUActivation::TanhInvCalculate(output);
#else
		Matrix c(output);
		for (size_t i = 0; i < c.GetRowCount() * c.GetColumnCount(); i++)
			c.SetValue(i, CalculateDerivate(c.GetValue(i)));
		return c;
#endif // USE_GPU

	}

	void CalculateInto(const Matrix& input, Matrix& target) override
	{
#if USE_GPU
		GPUActivation::TanhCalculate(input, target);
#else
		for (size_t i = 0; i < input.GetRowCount() * input.GetColumnCount(); i++)
			target.SetValue(i, Calculate(input.GetValue(i)));
#endif // USE_GPU

	}

	void CalculateDerivateInto(const Matrix& output, Matrix& target, float extra) override
	{
#if USE_GPU
		GPUActivation::TanhInvCalculate(output, target);
#else
		for (size_t i = 0; i < output.GetRowCount() * output.GetColumnCount(); i++)
			target.SetValue(i, CalculateDerivate(output.GetValue(i)));
#endif // USE_GPU

	}

	Tensor CalculateTensor(const Tensor& input) override
	{
		Tensor t(input);

		for (unsigned int i = 0; i < t.GetElementCount(); ++i)
			t.SetValue(i, Calculate(t.GetValue(i)));

		return t;
	}

	Tensor CalculateDerivateTensor(const Tensor& output, float extra) override
	{
		Tensor t(output);

		for (unsigned int i = 0; i < t.GetElementCount(); ++i)
			t.SetValue(i, CalculateDerivate(t.GetValue(i)));

		return t;
	}

	void CalculateInto(const Tensor& input, Tensor& target) override
	{
		for (unsigned int i = 0; i < input.GetElementCount(); ++i)
			target.SetValue(i, Calculate(input.GetValue(i)));
	}

	void CalculateDerivateInto(const Tensor& output, Tensor& target, float extra) override
	{
		for (unsigned int i = 0; i < output.GetElementCount(); ++i)
			target.SetValue(i, CalculateDerivate(output.GetValue(i)));
	}

private:
	static float Calculate(float a) {
		return tanh(a);
	}

	static float CalculateDerivate(float a) { return (float)(1 - pow(a, 2)); }

	TanhFunction() = default;
};
