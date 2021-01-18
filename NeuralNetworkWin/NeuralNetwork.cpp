#include <iostream>
#include "Matrix.h"
#include "MatrixMath.h"

#include <chrono>

int main()
{
	Matrix a(512, 512);
	Matrix b(512, 512);
	Matrix c(512, 512);

	MatrixMath::FillWithRandom(&a);
	MatrixMath::FillWithRandom(&b);

	std::chrono::steady_clock::time_point begin1 = std::chrono::steady_clock::now();
	Matrix* c1 = MatrixMath::SlowMultiply(&a, &b);
	std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
	std::cout << "Slow multiplication finished in: " << std::chrono::duration_cast<std::chrono::microseconds>(end1 - begin1).count() << "[µs]" << std::endl;


	std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();
	a.CopyToGPU();
	b.CopyToGPU();
	c.CopyToGPU();
	MatrixMath::Multiply(&a, &b, &c);
	c.CopyFromGPU();
	std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
	std::cout << "GPU multiplication finished in: " << std::chrono::duration_cast<std::chrono::microseconds>(end2 - begin2).count() << "[µs]" << std::endl;

	if (MatrixMath::IsEqual(c1, &c))
	{
		std::cout << "And the two outputs are equal!" << std::endl;
	}
	else
	{
		std::cout << "But there is an error in the GPU mul!" << std::endl;
	}

	return 0;
}
