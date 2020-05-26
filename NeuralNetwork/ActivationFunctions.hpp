#pragma once

#include "Matrix.h"
#include "MatrixMath.h"
#include <math.h>

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

class IdentityFunction : public ActivationFunction
{
public:
	IdentityFunction() {}
	~IdentityFunction() {}
	virtual Matrix* CalculateMatrix(Matrix* input) { Matrix* c(input); return c; }
	virtual Matrix* CalculateDerivateMatrix(Matrix* output, float extra = 0) { Matrix* c(output); MatrixMath::FillWith(c, 1); return c; }

	virtual void CalculateInto(Matrix* input, Matrix* target) { MatrixMath::Copy(input, target); }
	virtual void CalculateDerivateInto(Matrix* output, Matrix* target, float extra = 0) { MatrixMath::FillWith(target, 1); }
};

//TODO: Add Binary step

class Sigmoid : public ActivationFunction
{
public:
	Sigmoid() {}
	~Sigmoid() {}
	virtual Matrix* CalculateMatrix(Matrix* input)
	{
		Matrix* c(input);
		for (size_t i = 0; i < c->GetRowCount() * c->GetColumnCount(); i++)
			c->SetValue(i, Calculate(c->GetValue(i)));
		return c;
	}
	virtual Matrix* CalculateDerivateMatrix(Matrix* output, float extra = 0)
	{
		Matrix* c(output);
		for (size_t i = 0; i < c->GetRowCount() * c->GetColumnCount(); i++)
			c->SetValue(i, CalculateDerivate(c->GetValue(i)));
		return c;
	}

	virtual void CalculateInto(Matrix* input, Matrix* target)
	{
		for (size_t i = 0; i < input->GetRowCount() * input->GetColumnCount(); i++)
			target->SetValue(i, Calculate(input->GetValue(i)));
	}

	virtual void CalculateDerivateInto(Matrix* output, Matrix* target, float extra = 0)
	{
		for (size_t i = 0; i < output->GetRowCount() * output->GetColumnCount(); i++)
			target->SetValue(i, CalculateDerivate(output->GetValue(i)));
	}

private:
	float Calculate(float a) { return  1.0f / (1.0f + exp(-a)); }
	float CalculateDerivate(float a) { return a * (1 - a); }
};

class TanhFunction : public ActivationFunction
{
public:
	TanhFunction() {}
	~TanhFunction() {}
	virtual Matrix* CalculateMatrix(Matrix* input)
	{
		Matrix* c(input);
		for (size_t i = 0; i < c->GetRowCount() * c->GetColumnCount(); i++)
			c->SetValue(i, Calculate(c->GetValue(i)));
		return c;
	}
	virtual Matrix* CalculateDerivateMatrix(Matrix* output, float extra = 0)
	{
		Matrix* c(output);
		for (size_t i = 0; i < c->GetRowCount() * c->GetColumnCount(); i++)
			c->SetValue(i, CalculateDerivate(c->GetValue(i)));
		return c;
	}

	virtual void CalculateInto(Matrix* input, Matrix* target)
	{
		for (size_t i = 0; i < input->GetRowCount() * input->GetColumnCount(); i++)
			target->SetValue(i, Calculate(input->GetValue(i)));
	}

	virtual void CalculateDerivateInto(Matrix* output, Matrix* target, float extra = 0)
	{
		for (size_t i = 0; i < output->GetRowCount() * output->GetColumnCount(); i++)
			target->SetValue(i, CalculateDerivate(output->GetValue(i)));
	}

private:
	float Calculate(float a) {
		return ((exp(a) - exp(-a) / (exp(a) + exp(-a))));
	}

	float CalculateDerivate(float a) { return 1 - pow(a, 2); }
};
