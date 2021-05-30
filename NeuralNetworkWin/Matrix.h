#pragma once

#include <string>
#include "rapidjson/document.h"
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
	Matrix(const Matrix& c);

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
	 * @brief If the matrix is a vector, returns its size
	 * @return 0 if the matrix is 2D, the size of the vector otherwise
	*/
	unsigned int GetVectorSize();

	/**
	 * @brief Copies the values from another matrix.
	 * @param m The other matrix.
	*/
	void ReloadFromOther(const Matrix& m);

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

	void Reset(size_t rows, size_t columns);
	
	/**
	 * @brief Gets a single value from the matrix, similar to GetValue
	 * @param id The index of the value
	 * @return The value at the specified index
	*/
	float operator[](size_t id) const;

	Matrix& operator=(const Matrix& other);
	Matrix& operator=(Matrix&& other) noexcept;
	Matrix& operator+=(const Matrix& other);
	Matrix& operator-=(const Matrix& other);
	Matrix& operator*=(const Matrix& other);
	Matrix operator+(const Matrix& other) const;
	Matrix operator-(const Matrix& other)const;
	Matrix operator*(const Matrix& other) const;
	bool operator==(const Matrix& other) const;
	bool operator!=(const Matrix& other) const;

	Matrix& operator*=(float other);
	Matrix operator*(float other);

	inline size_t GetElementCount() const;

	Matrix GetSubMatrix(unsigned int startRow, unsigned int startColumn, unsigned int rowNum, unsigned int colNum) const;

	//std::ostream& operator<<(std::ostream& os, const Matrix& m);


private:
	float* Values;
	size_t Rows;
	size_t Columns;

	inline size_t RowColToPosition(size_t row, size_t col) const;

	float* GPUValues;

};