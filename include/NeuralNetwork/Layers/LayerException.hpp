#pragma once

#include <iostream>
#include <exception>

class LayerInputException : public std::exception
{
	const char* what() const throw()
	{
		return "Error code: 20 - Layer input not found";
	}
};

class LayerSizeException : public std::exception
{
	const char* what() const throw()
	{
		return "Error code: 21 - Matrix does not match layer size";
	}
};

class LayerTypeException : public std::exception
{
	const char* what() const throw()
	{
		return "Error code: 22 - Incorrect layer type";
	}
};