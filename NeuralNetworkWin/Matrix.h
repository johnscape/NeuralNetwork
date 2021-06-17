#pragma once

#include <string>
#include <iostream>

class Matrix
{
public:
	/**
	 * @brief Creates a 1x1 matrix, with a 0 element.
	*/
	Matrix();

	/**
	 * @brief Creates a matrix, in the shape of rows x columns.
	 * If the elements are not null, it'll copy the values into the matrix.
	 * It'll be filled with zeroes otherwise.
	 * @param rows The number of the rows
	 * @param columns The number of the columns
	 * @param elements An array, containing the values, to fill the matrix. Default: nullptr
	*/
	Matrix(size_t rows, size_t columns, float* elements = nullptr);

	/**
	 * @brief A copy constructor for the class
	 * @param c The other instance to copy
	*/
	Matrix(const Matrix& other);

	Matrix(Matrix&& other) noexcept;

	Matrix& operator=(const Matrix& other);

	Matrix& operator=(Matrix&& other) noexcept;

	~Matrix();

	/**
	 * @brief Returns the number of the columns.
	 * @return Number of the columns
	*/
	size_t GetColumnCount() const;

	/**
	 * @brief Returns the number of the rows.
	 * @return Number of the rows.
	*/
	size_t GetRowCount() const;

	/**
	 * @brief Returns with the value at the position of row and col. Return 0 if out of bounds.
	 * @param row The selected row
	 * @param col The selected column
	 * @return The value of the cell
	*/
	float GetValue(size_t row, size_t col) const;

	/**
	 * @brief Returns with the value at the position. Return 0 if out of bounds.
	 * @param pos The position in the matrix
	 * @return The value of the cell
	*/
	float GetValue(size_t pos) const;

	/**
	 * @brief Sets the value of the matrix at row and col.
	 * @param row The selected row
	 * @param col The selected column
	 * @param val The new value
	*/
	void SetValue(size_t row, size_t col, float val);

	/**
	 * @brief Sets the value of the matrix at the desired position.
	 * @param pos The matrix pos-th element
	 * @param val The new value
	*/
	void SetValue(size_t pos, float val);

	/**
	 * @brief Increments the value at the selected cell.
	 * @param row The selected row
	 * @param col The selected column
	 * @param val The value to be added
	*/
	void AdjustValue(size_t row, size_t col, float val);

	/**
	 * @brief Increments the value at the selected cell.
	 * @param pos The selected cell's index.
	 * @param val The value to be added
	*/
	void AdjustValue(size_t pos, float val);

	/**
	 * @brief Checks is the matrix is a vector
	 * @return True if the matrix is a vector
	*/
	bool IsVector() const;

	/**
	 * @brief If the matrix is a vector, returns its size
	 * @return 0 if the matrix is 2D, the size of the vector otherwise
	*/
	size_t GetVectorSize() const;

	/**
	 * @brief Copies the values from another matrix.
	 * @param m The other matrix.
	*/
	void ReloadFromOther(const Matrix& other);

	/**
	 * @brief Clears the matrix and creates a new empty one
	 * @param rows The new row number
	 * @param columns The new column number
	*/
	void Reset(size_t rows, size_t columns);

	/**
	 * @brief Copies the content of another matrix. The two matrices must be the same size, use ReladFromOtherInstead.
	 * @param from The other matrix to copy the values from.
	*/
	void Copy(const Matrix& from);

	/**
	 * @brief Fills the matrix with a selected value
	 * @param value The value to fill the matrix with
	*/
	void FillWith(float value);

	/**
	 * @brief Fills the matrix with random values between the selected boundaries
	 * @param min The minimum value (inclusive)
	 * @param max The maximum value (inclusive)
	*/
	void FillWithRandom(float min=-1, float max=1);

	/**
	 * @brief Loads the matrix from JSON data.
	 * @param data The JSON string or a file containing the JSON data.
	 * @param isFile Set it true if the data is in a file.
	*/
	void LoadFromJSON(const char* data, bool isFile = false);

	/**
	 * @brief Saves the matrix into a JSON string.
	 * @param fileName The file to save into. If don't want to save, leave it null.
	 * @return A string containing the JSON data of the matrix.
	*/
	std::string SaveToJSON(const char* fileName = nullptr);

	/**
	 * @brief If GPU is used, this function copies the values from the RAM to the GPU memory
	*/
	void CopyToGPU();

	/**
	 * @brief If GPU is used, this function copies the values from the GPU memory to the RAM
	*/
	void CopyFromGPU();

	/**
	 * @brief Returns the pointer for the GPU calculations
	 * @return The pointer of the GPU
	*/
	float* GetGPUValues();
	
	/**
	 * @brief Gets a single value from the matrix, similar to GetValue
	 * @param id The index of the value
	 * @return The value at the specified index
	*/
	float operator[](size_t id) const;

	//Operators

	Matrix& operator+=(const Matrix& other);
	Matrix& operator-=(const Matrix& other);
	Matrix& operator*=(const Matrix& other);
	Matrix operator+(const Matrix& other) const;
	Matrix operator-(const Matrix& other) const;
	Matrix operator*(const Matrix& other) const;
	bool operator==(const Matrix& other) const;
	bool operator!=(const Matrix& other) const;

	Matrix& operator*=(float other);
	Matrix operator*(float other);

	/**
	 * @brief Gets the number of elements in the matrix
	 * @return The product of the number of rows and columns.
	*/
	inline size_t GetElementCount() const;

	/**
	 * @brief Checks if two matrices are same sized.
	 * @param other The other matrix to compare to.
	 * @return A bool depending on the size of the matrices.
	*/
	inline bool IsSameSize(const Matrix& other) const;

	/**
	 * @brief Gets a sub-matrix based on the inputs
	 * @param startRow The first row to select
	 * @param startColumn The first column to select
	 * @param rowNum The number of rows to select
	 * @param colNum The number of columns to select
	 * @return A rowNum x colNum matrix, copied from this
	*/
	Matrix GetSubMatrix(size_t startRow, size_t startColumn, size_t rowNum, size_t colNum) const;

	/**
	 * @brief Creates a copy from a specific row of the matrix
	 * @param row The row to copy
	 * @return A matrix with one row. Throws MatrixIndexException if row parameter is too big.
	*/
	Matrix GetRowMatrix(size_t row) const;

	/**
	 * @brief Creates a copy from a specific column of the matrix
	 * @param col The column to copy
	 * @return A matrix with one column. Throws MatrixIndexException if col parameter is too big.
	*/
	Matrix GetColumnMatrix(size_t col) const;

	/**
	 * @brief Multiplies the matrix with another, elementwise
	 * @param other The matrix to multiply with
	*/
	void ElementwiseMultiply(const Matrix& other);

	Matrix OuterProduct(const Matrix& vector) const;

	float DotProcudt(const Matrix& vector) const;

	float Sum() const;

	/**
	 * @brief Transposes the matrix
	*/
	void Transpose();

	friend std::ostream& operator<<(std::ostream& os, const Matrix& m);

	/**
	 * @brief Multiplies two matrices together, elementwise. The matrices must have the same size.
	 * @param a The first matrix to multiply with
	 * @param b The second matrix to multiply with
	 * @return The product
	*/
	static Matrix ElementwiseMultiply(const Matrix& a, const Matrix& b);

	static Matrix Eye(unsigned int i);

	static Matrix Concat(const Matrix& a, const Matrix& b, unsigned int dim);

private:
	float* Values;
	size_t Rows;
	size_t Columns;

	inline size_t RowColToPosition(size_t row, size_t col) const;

	float* GPUValues;

	void TransposeArray(float* arr, unsigned int w, unsigned int h) const;
	void TransposeBlock(float* A, float* B, unsigned int lda, unsigned int ldb) const;

};