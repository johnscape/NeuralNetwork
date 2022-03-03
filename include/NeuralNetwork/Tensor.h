#pragma once

#include <vector>
#include <list>
#include <string>
#include "NeuralNetwork/Matrix.h"


class Tensor
{
public:
    /**
     * @brief Generates an empty 1x1 tensor.
     */
    Tensor();
    /**
     * @brief Generates a tensor with the shape defined in the correct parameter.
     * @param shape An array, containing the size of each dimension.
     * @param shapeCount The length of the shape parameter. Set to 0 to discover the value automatically.
     * @param values An array, containing the values of the tensor. Set it to nullptr to fill it with 0s.
     */
	explicit Tensor(unsigned int* shape, unsigned int shapeCount = 0, float* values = nullptr);
	/**
	 * @brief Creates a tensor with the shape defined in the correct parameter.
	 * @param shape A vector, containing the size of each dimension.
	 * @param values An array, containing the values of the tensor. Set it to nullptr to fill it with 0s.
	 */
	explicit Tensor(std::vector<unsigned int> shape, float* values = nullptr);
	/**
	 * @brief Creates a 2D tensor based on a matrix.
	 * @param mat The matrix to create from.
	 */
	explicit Tensor(const Matrix& mat);

	/**
	 * @brief Creates a 2D tensor by moving a matrix
	 * @param mat The matrix to move values from
	 */
	explicit Tensor(Matrix&& mat);

	/**
	 * @brief Creates the tensor by copying another
	 * @param other The tensor to copy
	 */
	Tensor(const Tensor& other);
	/**
	 * @brief The move constructor for the tensor
	 * @param other The tensor to move from
	 */
	Tensor(Tensor&& other) noexcept;

	~Tensor();

	/**
	 * @brief Compares the tensor's shape with another
	 * @param other The other tensor to compare to
	 * @return Returns true, if the shape of the two tensors are equal, false otherwise.
	 */
	bool IsSameShape(const Tensor& other) const;
	/**
	 * @brief Compares the tensor's first two dimension to a matrix's
	 * @param other The matrix to compare to
	 * @return True if the tensor's first two dimensions are the same as the matrix's row and column number
	 */
	bool IsSameShape(const Matrix& other) const;

	/**
	 * @brief Reshapes the tensor to a new shape. The number of elements must remain the same!
	 * @param newShape A vector containing the new shape of the tensor
	 */
	void Reshape(std::vector<unsigned int> newShape);
	void Reshape(unsigned int* newDimensions, unsigned int dimensionCount = 0);

	/**
	 * @brief Gets a value from the tensor at a defined position.
	 * @param position The position of the desired value
	 * @return The value. If the position does not exists, an error is thrown.
	 */
	float GetValue(unsigned int position) const;
	/**
	 * @brief Gets a value from the tensor at a defined position.
	 * @param position An array, describing the position of an element in the tensor (e.g.: 3, 2, 1)
	 * @return The value at the described position. If the value does not exists (the position is out of bounds), throws an error.
	 */
	float GetValue(unsigned int* position) const;

	/**
	 * @brief Sets a value at a specific position in the tensor
	 * @param pos The position of the value
	 * @param value The new value
	 */
	void SetValue(unsigned int pos, float value);

	/**
	 * @brief Adds a value to a specific cell in the tensor
	 * @param pos The cell to add to
	 * @param value The value to increment with
	 */
	void AdjustValue(unsigned int pos, float value);

	/**
	 * @brief Copies the given tensor into the current one
	 * @param other The other tensor to copy
	 */
	void ReloadFromOther(const Tensor& other);
	void ReloadFromOther(const Matrix& other);

	/**
	 * @brief Copies the values from a same sized tensor
	 * @param other The tensor to copy from
	 */
	void Copy(const Tensor& other);

	/**
	 * @brief Gets the tensor's shape
	 * @return A vector containing the shape of the tensor
	 */
	std::vector<unsigned int> GetShape() const;
	/**
	 * @brief A function to convert the shape into a printable string
	 * @return The tensor's shape in a string
	 */
	std::string GetShapeAsString() const;

	/**
	 * @brief Returns the size of the tensor at a defined dimension
	 * @param i The dimension to check
	 * @return The size of a specific dimension
	 */
	unsigned int GetShapeAt(unsigned int i) const;

	/**
	 * @brief Gets the first nxk values of the tensor and converts it into a matrix
	 * @return A matrix containing the first nxk values of the tensor
	 */
	Matrix FirstMatrix() const;
	/**
	 * @brief Converts the tensor into a list of matrices.
	 * @return A list, containing the matrices that represents the tensor
	 */
	std::list<Matrix> ToMatrixList() const;

	/**
	 * @brief Gets the Nth matrix from the tensor
	 * @param n The number of the wanted matrix.
	 * @param mat The matrix to copy values into.
	 */
	void GetNthMatrix(unsigned int n, Matrix* mat = nullptr);

	/**
	 * @brief Converts a specific row into a matrix
	 * @param matrix The matrix with the wanted row
	 * @param row The wanted row number
	 * @return A row matrix of the wanted row
	 */
	Matrix GetRowMatrix(unsigned int matrix, unsigned int row) const;

	Matrix ToMatrixByRows() const;

	/**
	 * @brief Gets the number of elements in the tensor
	 * @return The number of elements
	 */
	unsigned int GetElementCount() const;

	/**
	 * @brief Gets the number of matrices in the tensor
	 * @return The number of matrices
	 */
	unsigned int GetMatrixCount() const;

	float Sum() const;

	/**
	 * @brief Sets every value of the tensor to a specified value
	 * @param value The desired value
	 */
	void FillWith(float value);
	/**
	 * @brief Sets each value in the tensor to a random number, between the min and max values (min and max are included)
	 * @param min The minimum random value
	 * @param max The maximum random value
	 */
	void FillWithRandom(float min = -1, float max = 1);

	//matrix operators
	Tensor operator+(const Matrix& other) const;
	Tensor operator-(const Matrix& other) const;
	Tensor operator*(const Matrix& other) const;

	Tensor& operator+=(const Matrix& other);
	Tensor& operator-=(const Matrix& other);
	Tensor& operator*=(const Matrix& other);

	bool operator==(const Matrix& other) const;
	bool operator!=(const Matrix& other) const;

	//tensor operators
	Tensor operator+(const Tensor& other) const;
	Tensor operator-(const Tensor& other) const;
	Tensor operator*(const Tensor& other) const;

	Tensor& operator+=(const Tensor& other);
	Tensor& operator-=(const Tensor& other);
	Tensor& operator*=(const Tensor& other);

	Tensor& operator=(const Tensor& other);
	Tensor& operator=(Tensor&& other) noexcept ;

	bool operator==(const Tensor& other) const;
	bool operator!=(const Tensor& other) const;

	friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

private:
	std::vector<unsigned int> Shape;
	float* Values;
	size_t ElementCount;

	unsigned int CoordinateToPos(unsigned int* coord) const;
	unsigned int CoordinateToPos(std::vector<unsigned int> coord) const;

	Tensor& RowBasedAddition(const Matrix& mat, bool local);
};