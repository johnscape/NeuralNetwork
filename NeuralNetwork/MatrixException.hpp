#pragma once

#include <iostream>
#include <exception>

class MatrixException : public std::exception
{
	const char* what() const throw()
	{
		return "Error code: 10 - Matrix error";
	}
};

class MatrixSizeException : public std::exception
{
	const char* what() const throw()
	{
		return "Error code: 11 - Matrix size is incorrect";
	}
};

class MatrixIndexException : public std::exception
{
	const char* what() const throw()
	{
		return "Error code: 12 - Matrix index is out of bounds";
	}
};