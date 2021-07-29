#include <catch2/catch.hpp>

#include "../include/NeuralNetwork/Matrix.h"
#include "../include/NeuralNetwork/Tensor.h"
#include "../include/NeuralNetwork/TensorException.hpp"

SCENARIO("Creating tensors", "[tensor]")
{
	GIVEN("an empty tensor")
	{
		Tensor t;

		WHEN("checking the shape of the tensor")
		{
			THEN("the shape size is 0")
			{
				REQUIRE(t.GetShape().size() == 0);
			}
		}
	}
	GIVEN("a 3x2x4 tensor")
	{
		Tensor t({ 3, 2, 4 });
		WHEN("checking shape")
		{
			THEN("the size of the shape is 3")
			{
				REQUIRE(t.GetShape().size() == 3);
			}
			THEN("the shape is 3x2x4")
			{
				REQUIRE(t.GetShape()[0] == 3);
				REQUIRE(t.GetShape()[1] == 2);
				REQUIRE(t.GetShape()[2] == 4);
			}
		}
	}
}
SCENARIO("comparing tensors", "[tensor]")
{
	GIVEN("two tensors with the same shape")
	{
		Tensor t1({ 1, 2, 3 });
		Tensor t2({ 1, 2, 3 });
		WHEN("comparing shapes")
		{
			THEN("the tensors have the same shape")
			{
				REQUIRE(t1.IsSameShape(t2));
			}
		}
	}
	GIVEN("two tensor with different shapes")
	{
		Tensor t1({ 1, 2, 3 });
		Tensor t2({ 3, 2, 1 });

		WHEN("comparing the shape of the tensors")
		{
			THEN("the two shape is not the same")
			{
				REQUIRE_FALSE(t1.IsSameShape(t2));
			}
		}
	}
}

SCENARIO("reshaping tensors", "[tensor]")
{
	GIVEN("a 3x3x1 tensor")
	{
		Tensor t({ 3, 3, 1 });
		WHEN("reshaping the tensor to 1x3x3")
		{
			t.Reshape({ 1, 3, 3 });
			THEN("the tensor's shape's size is 3")
			{
				REQUIRE(t.GetShape().size() == 3);
			}
			THEN("the tensor's shape is 1x3x3")
			{
				REQUIRE(t.GetShape()[0] == 1);
				REQUIRE(t.GetShape()[1] == 3);
				REQUIRE(t.GetShape()[2] == 3);
			}
		}
		WHEN("reshaping to 1x4x1")
		{
			THEN("the tensor raises an error")
			{
				REQUIRE_THROWS_AS(t.Reshape({ 1, 4, 1 }), TensorShapeException);
			}
		}
	}
}

SCENARIO("converting tensor to matrix", "[tensor][matrix]")
{
	GIVEN("a 3x5x4x1 tensor")
	{
		Tensor t({ 3, 5, 4, 1 });
		WHEN("getting the first matrix")
		{
			Matrix m = t.FirstMatrix();

			THEN("the first matrix is 3x5")
			{
				REQUIRE(m.GetRowCount() == 3);
				REQUIRE(m.GetColumnCount() == 5);
			}
		}
		WHEN("converting to matrix list")
		{
			std::list<Matrix> lst = t.ToMatrixList();

			THEN("the list lenght is 4")
			{
				REQUIRE(lst.size() == 4);
			}
			THEN("every matrix is 3x5")
			{
				std::list<Matrix>::iterator it = lst.begin();
				for (; it != lst.end(); it++)
				{
					REQUIRE(it->GetRowCount() == 3);
					REQUIRE(it->GetColumnCount() == 5);
				}
			}
		}
	}
}