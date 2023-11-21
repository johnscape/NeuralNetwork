#pragma once

#include <vector>
#include <list>
#include <string>
#include "NeuralNetwork/TempMatrix.h"

class Matrix;

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
	void Reshape(const std::vector<unsigned int>& newShape);
	void Reshape(unsigned int* newDimensions, unsigned int dimensionCount = 0);

	/**
	 * @brief Gets a value from the tensor at a defined position.
	 * @param position The position of the desired value
	 * @return The value. If the position does not exists, an error is thrown.
	 */
	float GetValue(unsigned int position) const;

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
	 * @brief Gets the n-th matrix in order into a temp matrix
	 * @param n The matrix to select
	 * @return The selected matrix as a temp matrix
	 */
	TempMatrix GetNthTempMatrix(unsigned int n) const;

	/**
	 * @brief Converts a specific row into a matrix
	 * @param matrix The matrix with the wanted row
	 * @param row The wanted row number
	 * @return A row matrix of the wanted row
	 */
	Matrix GetRowMatrix(unsigned int matrix, unsigned int row) const;

	/**
	 * @brief Returns the tensor as a temp matrix, with all of the matrices converted into new rows
	 * @return A temp matrix with extra rows as matrices
	 */
	TempMatrix ToMatrixByRows() const;

    void CopyPartTo(Tensor& target, unsigned int startLocal, unsigned int startTarget, unsigned int count) const;
    void CopyPartTo(Matrix& target, unsigned int startLocal, unsigned int startTarget, unsigned int count) const;

	/**
	 * @brief Copies a matrix into a specified position in the tensor
	 * @param n The matrix to overwrite
	 * @param mat The matrix to copy the values from
	 */
	void LoadMatrix(unsigned int n, Matrix* mat);

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

	/**
	 * @brief Adds the values in the tensor together
	 * @return The sum of values inside of the tensor
	 */
	float Sum() const;

	/**
	 * @brief Removes the 1 dimensions from the tensor
	 */
	void Squeeze();

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

	/**
	 * @brief Loads the tensor from JSON data.
	 * @param data The JSON string or a file containing the JSON data.
	 * @param isFile Set it true if the data is in a file.
	*/
	void LoadFromJSON(const char* data, bool isFile = false);

	/**
	 * @brief Loads the tensor from JSON data.
	 * @param jsonValue rapidjson::Value type containing the data of the tensor
	 */
	void LoadFromJSON(rapidjson::Value& jsonValue);

	/**
	 * @brief Saves the tensor into a JSON string.
	 * @param fileName The file to save into. If don't want to save, leave it null.
	 * @return A string containing the JSON data of the tensor.
	*/
	std::string SaveToJSON(const char* fileName = nullptr) const;

	/**
	 * @brief Saves the tensor into a JSON type
	 * @return The JSONified tensor
	 */
	rapidjson::Value SaveToJSONObject(rapidjson::Document& document) const;

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

    void CopyToGPU();
    void CopyFromGPU();
    void CopyFromOtherGPU(const Tensor& other);

    float* GetGPUValues();
    float* GetConstGPUValues() const;

    friend class Matrix;

private:
	std::vector<unsigned int> Shape;
	float* Values;
    float* GPUValues;
	size_t ElementCount;

	unsigned int CoordinateToPos(unsigned int* coord) const;
	unsigned int CoordinateToPos(std::vector<unsigned int> coord) const;

	Tensor& RowBasedAddition(const Matrix& mat, bool local);

    void MallocGPU();
    void FreeGPU();
};