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

private:
	std::vector<Matrix> matrices;
	unsigned int Rows;
	unsigned int Columns;
	unsigned int Channels;
	unsigned int TimeSteps;
	unsigned int Batches;
};

