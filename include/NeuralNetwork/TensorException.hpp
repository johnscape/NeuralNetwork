#pragma once

#include <iostream>
#include <exception>

class TensorException : public std::exception
{
	const char* what() const throw()
	{
		return "Error code: 20 - Tensor error";
	}
};

class TensorShapeException : public std::exception
{
	const char* what() const throw()
	{
		return "Error code: 21 - Invalid tensor dimensions";
	}
};

class TensorIndexException : public std::exception
{
	const char* what() const throw()
	{
		return "Error code: 22 - Invalid index number";
	}
};