#pragma once

#include <string>
#include "rapidjson/document.h"

class Matrix
{
public:
	/// <summary>
	/// Creates a 1x1 matrix, with a 0 element.
	/// </summary>
	Matrix();

	/// <summary>
	/// Creates a matrix, in the shape of rows x columns.
	/// If the elements are not null, it'll copy the values into the matrix.
	/// It'll be filled with zeroes otherwise.
	/// </summary>
	/// <param name="rows">The number of the rows</param>
	/// <param name="columns">The number of the columns</param>
	/// <param name="elements">An array, containing the values, to fill the matrix. Default: nullptr</param>
	Matrix(size_t rows, size_t columns, float* elements = nullptr);

	/// <summary>
	/// A copy constructor for the class
	/// </summary>
	/// <param name="c">The other instance to copy</param>
	Matrix(const Matrix& c);

	~Matrix();

	/// <summary>
	/// Returns the number of the columns.
	/// </summary>
	/// <returns>Number of the columns</returns>
	size_t GetColumnCount() const;

	/// <summary>
	/// Returns the number of the rows.
	/// </summary>
	/// <returns>Number of the rows.</returns>
	size_t GetRowCount() const;



	/// <summary>
	/// Returns with the value at the position of row and col. Return 0 if out of bounds.
	/// </summary>
	/// <param name="row"></param>
	/// <param name="col"></param>
	/// <returns></returns>
	float GetValue(size_t row, size_t col) const;

	/// <summary>
	///  Returns with the value at the position. Return 0 if out of bounds.
	/// </summary>
	/// <param name="pos">The position in the matrix</param>
	/// <returns></returns>
	float GetValue(size_t pos) const;



	/// <summary>
	/// Sets the value of the matrix at row and col.
	/// </summary>
	/// <param name="row">The selected row</param>
	/// <param name="col">The selected column</param>
	/// <param name="val">The new value</param>
	void SetValue(size_t row, size_t col, float val);

	/// <summary>
	/// Sets the value of the matrix at the desired position.
	/// </summary>
	/// <param name="pos">The matrix pos-th element</param>
	/// <param name="val">The new value</param>
	void SetValue(size_t pos, float val);



	/// <summary>
	/// Adjust the value at the selected cell.
	/// </summary>
	/// <param name="row">The selected row</param>
	/// <param name="col">The selected column</param>
	/// <param name="val">The value to be added</param>
	void AdjustValue(size_t row, size_t col, float val);

	/// <summary>
	/// Adjust the value at the selected cell.
	/// </summary>
	/// <param name="pos">The selected position</param>
	/// <param name="val">The value to be added</param>
	void AdjustValue(size_t pos, float val);



	/// <summary>
	/// Returns the selected value
	/// </summary>
	/// <param name="id">The position</param>
	/// <returns></returns>
	float& operator[](size_t id);

	unsigned int GetVectorSize();

	void ReloadFromOther(Matrix* m);

	void LoadFromJSON(const char* data, bool isFile = false);
	std::string SaveToJSON(const char* fileName = nullptr);

private:
	float* Values;
	size_t Rows;
	size_t Columns;

	size_t MaxValue;

	inline size_t RowColToPosition(size_t row, size_t col) const;
};