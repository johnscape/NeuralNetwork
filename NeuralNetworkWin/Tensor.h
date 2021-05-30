#pragma once
#include "Matrix.h"
#include <vector>

/* TODO: Implement tensor class
* This class will act as a multi-dimensional matrix
* 0th dimension: rows
* 1st dimension: cols
* 2nd dimension: channels
* 3rd dimension: time-steps
* 4th dimension: batch
*/
class Tensor
{
public:
	Tensor();
	Tensor(unsigned int rows, unsigned int cols, float* values = nullptr);
	Tensor(unsigned int rows, unsigned int cols, float* values = nullptr, unsigned int channels = 1, unsigned int times = 1, unsigned int batch = 1);
	Tensor(const Tensor& other);

	Tensor operator+(const Tensor& other);
	Tensor operator-(const Tensor& other);
	Tensor operator*(const Tensor& other);
	Tensor& operator=(const Tensor& other);
	Tensor& operator=(Tensor&& other) noexcept;
	Tensor& operator+=(const Tensor& other);
	Tensor& operator-=(const Tensor& other);
	Tensor& operator*=(const Tensor& other);

	bool operator==(const Tensor& other);
	bool operator!=(const Tensor& other);

	Matrix& GetMatrixAt(unsigned int channel = 0, unsigned int timestep = 0, unsigned int batch = 0);

	void AddChannel();
	void AddTimeStep();
	void AddBatch();

	void SetChannels(unsigned int count);
	void SetTimeSteps(unsigned int count);
	void SetBatchCount(unsigned int count);

	unsigned int GetRowCount() const;
	unsigned int GetColumnCount() const;
	unsigned int GetChannelCount() const;
	unsigned int GetTimeSteps() const;
	unsigned int GetBatchNumber() const;

private:
	std::vector<Matrix> matrices;
	unsigned int Rows;
	unsigned int Columns;
	unsigned int Channels;
	unsigned int TimeSteps;
	unsigned int Batches;
};

