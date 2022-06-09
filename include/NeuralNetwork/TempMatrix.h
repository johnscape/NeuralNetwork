#pragma once

#include "Matrix.h"

/**
 * @brief A matrix class used to remove memory allocations from one-time matrices
 */
class TempMatrix : public Matrix
{
public:
	TempMatrix();
	TempMatrix(size_t rows, size_t cols, float* values);
	~TempMatrix();

	Matrix ToMatrix();
	virtual void Transpose();

	//deleted functions
	void SetValue(size_t row, size_t col, float val) = delete;
	void SetValue(size_t pos, float val) = delete;
	void AdjustValue(size_t row, size_t col, float val) = delete;
	void AdjustValue(size_t pos, float val) = delete;
	void ReloadFromOther(const Matrix& other) = delete;
	void Reset(size_t rows, size_t columns) = delete;
	void Copy(const Matrix& from) = delete;
	void FillWith(float value) = delete;
	void FillWithRandom(float min, float max) = delete;
	void LoadFromJSON(const char* data, bool isFile) = delete;
	void CopyToGPU() = delete;
	void CopyFromGPU() = delete;
	Matrix& operator+=(const Matrix& other) = delete;
	Matrix& operator-=(const Matrix& other) = delete;
	Matrix& operator*=(const Matrix& other) = delete;
	Matrix& operator*=(float other) = delete;
	Matrix operator*(float other) = delete;
	void ElementwiseMultiply(const Matrix& other) = delete;
	void Clamp(float min, float max) = delete;
	void RoundToInt() = delete;
	void Pad(unsigned int top, unsigned int left, unsigned int bottom, unsigned int right, PadType type, float value) = delete;
	void ToSquare() = delete;
	void Rotate(unsigned int times) = delete;
	void Normalize(float maxValue) = delete;
	void PowerSelf(unsigned int p) = delete;

	//operators
	Matrix operator+(const Matrix& other) const;
	Matrix operator-(const Matrix& other) const;
	Matrix operator*(const Matrix& other) const;
	bool operator==(const Matrix& other) const;
	bool operator!=(const Matrix& other) const;

};