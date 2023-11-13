#pragma once

#include "Matrix.h"
#include <cmath>
#include <memory>
#include "Constants.h"

#define ACTIVATION_SIGMOID &Sigmoid::GetInstance()
#define ACTIVATION_TANH &TanhFunction::GetInstance()
#define ACTIVATION_LINEAR &IdentityFunction::GetInstance()

#if USE_GPU==USING_CUDA
#include "GPUActivation.cuh"
#endif // USE_GPU

enum class ActivationFunctionType
{
	IDENTITY,
	BINARYSTEP,
	SIGMOID,
	TANH,
	RELU
};

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

	virtual ActivationFunctionType GetActivationFunctionType() = 0;
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

	virtual ActivationFunctionType GetActivationFunctionType() {return ActivationFunctionType::IDENTITY;}

private:
	IdentityFunction() = default;

};

class BinaryStep : public ActivationFunction
{
public:
	static BinaryStep& GetInstance()
	{
		static BinaryStep func;
		return func;
	}

	~BinaryStep() = default;
	Matrix CalculateMatrix(const Matrix& input) override
	{
		Matrix c(input.GetRowCount(), input.GetColumnCount());
		for (size_t i = 0; i < input.GetElementCount(); ++i)
			c.SetValue(i, input.GetValue(i) > 0 ? 1 : 0);
		return c;
	}

	Matrix CalculateDerivateMatrix(const Matrix& output, float extra) override
	{
		Matrix c(output);
		c.FillWith(0);
		return c;
	}

	void CalculateInto(const Matrix& input, Matrix& target) override
	{
		for (size_t i = 0; i < input.GetElementCount(); ++i)
			target.SetValue(i, input.GetValue(i) > 0 ? 1 : 0);
	}

	void CalculateDerivateInto(const Matrix& output, Matrix& target, float extra) override
	{
		target.FillWith(0);
	}

	Tensor CalculateTensor(const Tensor& input) override
	{
		Tensor t(input);
		for (unsigned int i = 0; i < t.GetElementCount(); i++)
			t.SetValue(i, t.GetValue(i) > 0 ? 1 : 0);
		return t;
	}

	Tensor CalculateDerivateTensor(const Tensor& output, float extra) override
	{
		Tensor t(output.GetShape());
		t.FillWith(0);
		return t;
	}

	void CalculateInto(const Tensor& input, Tensor& target) override
	{
		for (unsigned int i = 0; i < input.GetElementCount(); i++)
			target.SetValue(i, input.GetValue(i) > 0 ? 1 : 0);
	}

	void CalculateDerivateInto(const Tensor& output, Tensor& target, float extra) override
	{
		target.FillWith(0);
	}

	virtual ActivationFunctionType GetActivationFunctionType() {return ActivationFunctionType::BINARYSTEP;}

private:
	BinaryStep() = default;
};

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

	~Sigmoid() = default;
	Matrix CalculateMatrix(const Matrix& input) override
	{
#if USE_GPU==USING_CUDA
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
#if USE_GPU==USING_CUDA
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
#if USE_GPU==USING_CUDA
		GPUActivation::SigmoidCalculate(input, target);
#else
		for (size_t i = 0; i < input.GetRowCount() * input.GetColumnCount(); i++)
			target.SetValue(i, Calculate(input.GetValue(i)));
#endif // USE_GPU

	}

	void CalculateDerivateInto(const Matrix& output, Matrix& target, float extra) override
	{
#if USE_GPU==USING_CUDA
		GPUActivation::SigmoidInvCalculate(output, target);
#else
		for (size_t i = 0; i < output.GetRowCount() * output.GetColumnCount(); i++)
			target.SetValue(i, CalculateDerivate(output.GetValue(i)));
#endif // USE_GPU

	}

	Tensor CalculateTensor(const Tensor& input) override
	{
#if USE_GPU==USING_CUDA
        return GPUActivation::SigmoidCalculate(input);
#else
		Tensor t(input);

		for (unsigned int i = 0; i < t.GetElementCount(); ++i)
			t.SetValue(i, Calculate(t.GetValue(i)));

		return t;
#endif
	}

	Tensor CalculateDerivateTensor(const Tensor& output, float extra) override
	{
#if USE_GPU==USING_CUDA
        return GPUActivation::SigmoidInvCalculate(output);
#else
		Tensor t(output);

		for (unsigned int i = 0; i < t.GetElementCount(); ++i)
			t.SetValue(i, CalculateDerivate(t.GetValue(i)));

		return t;
#endif
	}

	void CalculateInto(const Tensor& input, Tensor& target) override
	{
#if USE_GPU==USING_CUDA
        GPUActivation::SigmoidCalculate(input, target);
#else
		for (unsigned int i = 0; i < input.GetElementCount(); ++i)
			target.SetValue(i, Calculate(input.GetValue(i)));
#endif
	}

	void CalculateDerivateInto(const Tensor& output, Tensor& target, float extra) override
	{
#if USE_GPU==USING_CUDA
        GPUActivation::SigmoidInvCalculate(output, target);
#else
		for (unsigned int i = 0; i < output.GetElementCount(); ++i)
			target.SetValue(i, CalculateDerivate(output.GetValue(i)));
#endif
	}

	virtual ActivationFunctionType GetActivationFunctionType() {return ActivationFunctionType::SIGMOID;}

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

	~TanhFunction() = default;
	Matrix CalculateMatrix(const Matrix& input) override
	{
#if USE_GPU==USING_CUDA
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
#if USE_GPU==USING_CUDA
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
#if USE_GPU==USING_CUDA
		GPUActivation::TanhCalculate(input, target);
#else
		for (size_t i = 0; i < input.GetRowCount() * input.GetColumnCount(); i++)
			target.SetValue(i, Calculate(input.GetValue(i)));
#endif // USE_GPU

	}

	void CalculateDerivateInto(const Matrix& output, Matrix& target, float extra) override
	{
#if USE_GPU==USING_CUDA
		GPUActivation::TanhInvCalculate(output, target);
#else
		for (size_t i = 0; i < output.GetRowCount() * output.GetColumnCount(); i++)
			target.SetValue(i, CalculateDerivate(output.GetValue(i)));
#endif // USE_GPU

	}

	Tensor CalculateTensor(const Tensor& input) override
	{
#if USE_GPU==USING_CUDA
        return GPUActivation::TanhCalculate(input);
#else
		Tensor t(input);

		for (unsigned int i = 0; i < t.GetElementCount(); ++i)
			t.SetValue(i, Calculate(t.GetValue(i)));

		return t;
#endif
	}

	Tensor CalculateDerivateTensor(const Tensor& output, float extra) override
	{
#if USE_GPU==USING_CUDA
        return GPUActivation::TanhInvCalculate(output);
#else
		Tensor t(output);

		for (unsigned int i = 0; i < t.GetElementCount(); ++i)
			t.SetValue(i, CalculateDerivate(t.GetValue(i)));

		return t;
#endif
	}

	void CalculateInto(const Tensor& input, Tensor& target) override
	{
#if USE_GPU==USING_CUDA
        GPUActivation::TanhCalculate(input, target);
#else
		for (unsigned int i = 0; i < input.GetElementCount(); ++i)
			target.SetValue(i, Calculate(input.GetValue(i)));
#endif
	}

	void CalculateDerivateInto(const Tensor& output, Tensor& target, float extra) override
	{
#if USE_GPU==USING_CUDA
        GPUActivation::TanhInvCalculate(output, target);
#else
		for (unsigned int i = 0; i < output.GetElementCount(); ++i)
			target.SetValue(i, CalculateDerivate(output.GetValue(i)));
#endif
	}

	virtual ActivationFunctionType GetActivationFunctionType() {return ActivationFunctionType::TANH;}

private:
	static float Calculate(float a) {
		return tanh(a);
	}

	static float CalculateDerivate(float a) { return (float)(1 - pow(a, 2)); }

	TanhFunction() = default;
};

class RELU : public ActivationFunction
{
public:
	static RELU& GetInstance()
	{
		static RELU func;
		return func;
	}

	virtual Matrix CalculateMatrix(const Matrix& input)
	{
		Matrix c(input);
		for (unsigned int i = 0; i < c.GetElementCount(); ++i) {
			float v = c.GetValue(i);
			if (v <= 0)
				c.SetValue(i, 0);
			else
				c.SetValue(i, v);
		}
		return c;
	}

	virtual Matrix CalculateDerivateMatrix(const Matrix& output, float extra) {
		Matrix c(output);
		for (unsigned int i = 0; i < c.GetElementCount(); ++i)
		{
			float v = c.GetValue(i);
			if (v <= 0)
				c.SetValue(i, 0);
			else
				c.SetValue(i, 1);
		}
		return c;
	}

	Matrix CalculateDerivateMatrix(const Matrix& output) {
		return CalculateDerivateMatrix(output, 0);
	}

	virtual void CalculateInto(const Matrix& input, Matrix& target) {
		for (unsigned int i = 0; i < input.GetElementCount(); ++i) {
			float v = input.GetValue(i);
			if (v <= 0)
				target.SetValue(i, 0);
			else
				target.SetValue(i, v);
		}
	}

	virtual void CalculateDerivateInto(const Matrix& output, Matrix& target, float extra) {
		for (unsigned int i = 0; i < output.GetElementCount(); ++i)
		{
			if (output.GetValue(i) <= 0)
				target.SetValue(i, 0);
			else
				target.SetValue(i, 1);
		}
	}

	virtual Tensor CalculateTensor(const Tensor& input) {
		Tensor c(input);
		for (unsigned int i = 0; i < c.GetElementCount(); ++i) {
			float v = c.GetValue(i);
			if (v <= 0)
				c.SetValue(i, 0);
			else
				c.SetValue(i, v);
		}
		return c;
	}

	virtual Tensor CalculateDerivateTensor(const Tensor& output, float extra) {
		Tensor c(output);
		for (unsigned int i = 0; i < c.GetElementCount(); ++i)
		{
			float v = c.GetValue(i);
			if (v <= 0)
				c.SetValue(i, 0);
			else
				c.SetValue(i, 1);
		}
		return c;
	}
	Tensor CalculateDerivateTensor(const Tensor& output) {
		return CalculateDerivateTensor(output, 0);
	}

	virtual void CalculateInto(const Tensor& input, Tensor& target) {
		for (unsigned int i = 0; i < input.GetElementCount(); ++i) {
			float v = input.GetValue(i);
			if (v <= 0)
				target.SetValue(i, 0);
			else
				target.SetValue(i, v);
		}
	}

	virtual void CalculateDerivateInto(const Tensor& output, Tensor& target, float extra) {
		for (unsigned int i = 0; i < output.GetElementCount(); ++i)
		{
			if (output.GetValue(i) <= 0)
				target.SetValue(i, 0);
			else
				target.SetValue(i, 1);
		}
	}

	void CalculateDerivateInto(const Tensor& output, Tensor& target) {
		CalculateDerivateInto(output, target, 0);
	}

	virtual ActivationFunctionType GetActivationFunctionType() {return ActivationFunctionType::RELU;}

private:
	RELU() = default;
};

class ActivationFunctionLibrary
{
public:

	static ActivationFunction* GetActivationFunction(ActivationFunctionType type)
	{
		if (type == ActivationFunctionType::IDENTITY)
			return &IdentityFunction::GetInstance();
		if (type == ActivationFunctionType::BINARYSTEP)
			return &BinaryStep::GetInstance();
		if (type == ActivationFunctionType::SIGMOID)
			return &Sigmoid::GetInstance();
		if (type == ActivationFunctionType::TANH)
			return &Sigmoid::GetInstance();
		if (type == ActivationFunctionType::RELU)
			return &RELU::GetInstance();
	}
};