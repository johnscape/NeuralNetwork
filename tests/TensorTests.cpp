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
	GIVEN("a vector of the following values: 3, 2,8")
	{
	    std::vector<unsigned int> values = {3, 2, 8};

	    WHEN("creating a tensor from this vector")
	    {
	        Tensor t(values);
	        THEN("the shape of the tensor is 3x2x8")
	        {
                REQUIRE(t.GetShape().size() == 3);
                REQUIRE(t.GetShape()[0] == 3);
                REQUIRE(t.GetShape()[1] == 2);
                REQUIRE(t.GetShape()[2] == 8);
	        }
	    }
	}
	GIVEN("an array of [2, 2] and an array of [0, 1, 2, 3]")
	{
	    unsigned int dims[2] = {2, 2};
	    float vals[4] = {0, 1, 2, 3};

	    WHEN("creating a tensor from these values")
	    {
	        Tensor t(dims, 2, vals);

	        THEN("the dimensions are 2x2")
	        {
                REQUIRE(t.GetShape().size() == 2);
                REQUIRE(t.GetShape()[0] == 2);
                REQUIRE(t.GetShape()[1] == 2);
	        }
	        THEN("the values are the same as in the value array")
	        {
                for (int i = 0; i < 4; ++i)
                    REQUIRE(t.GetValue(i) == i);
	        }
	    }
	}
	GIVEN("a 3x3 matrix, filled with 5")
	{
	    Matrix mat(3, 3);
	    mat.FillWith(5);

	    WHEN("creating a tensor from this matrix")
	    {
	        Tensor t(mat);

	        THEN("the tensor's shape is 3x3")
	        {
                REQUIRE(t.GetShape().size() == 2);
                REQUIRE(t.GetShape()[0] == 3);
                REQUIRE(t.GetShape()[1] == 3);
	        }
	        THEN("every value in the tensor is 5")
	        {
                for (int i = 0; i < 9; ++i)
                    REQUIRE(t.GetValue(i) == 5);
	        }
	    }
	}
	GIVEN("a 5x3x4 tensor")
	{
	    Tensor t1({5, 3, 4});
	    WHEN("calling the move constructor")
	    {
	        Tensor t2(std::move(t1));
	        THEN("the new tensor's shape is 5x3x4")
	        {
	            REQUIRE(t2.GetShape().size() == 3);
	            REQUIRE(t2.GetShape()[0] == 5);
	            REQUIRE(t2.GetShape()[1] == 3);
	            REQUIRE(t2.GetShape()[2] == 4);
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

SCENARIO("modifying the tensors", "[tensor]")
{
	GIVEN("a 3x4x5 tensor")
	{
		Tensor t({3, 4, 5});

		WHEN("filling the tensor with 4")
		{
			t.FillWith(4);
			THEN("every value is 4")
			{
				for (int i = 0; i < 3*4*5; ++i)
					REQUIRE(t.GetValue(i) == 4);
			}
		}
		WHEN("filling the tensor with random values")
		{
			t.FillWithRandom();
			THEN("every value is between -1 and 1")
			{
				for (int i = 0; i < 3*4*5; ++i)
				{
					REQUIRE(t.GetValue(i) <= 1);
					REQUIRE(t.GetValue(i) >= -1);
				}
			}
		}
	}
}

SCENARIO("using tensor-matrix arithmetics", "[tensor][math]")
{
	GIVEN("a 3x2x5 tensor filled with 1, and a 3x2 matrix filled with 5")
	{
		Tensor t({3, 2, 5});
		Matrix m(3, 2);

		t.FillWith(1);
		m.FillWith(5);

		WHEN("adding the matrix to the tensor")
		{
			Tensor res;
			res = t + m;

			THEN("every value is 6")
			{
				for (int i = 0; i < res.GetElementCount(); ++i)
					REQUIRE(res.GetValue(i) == 6);
			}
		}
		WHEN("subtracting the matrix from the tensor")
		{
			Tensor res;
			res = t - m;

			THEN("every value is -4")
			{
				for (int i = 0; i < res.GetElementCount(); ++i)
					REQUIRE(res.GetValue(i) == -4);
			}
		}

		WHEN("incrementing the tensor with the matrix")
		{
			t += m;

			THEN("every value is 6")
			{
				for (int i = 0; i < t.GetElementCount(); ++i)
					REQUIRE(t.GetValue(i) == 6);
			}
		}

		WHEN("decrementing the tensor with the matrix")
		{
			t -= m;

			THEN("every value is -4")
			{
				for (int i = 0; i < t.GetElementCount(); ++i)
					REQUIRE(t.GetValue(i) == -4);
			}
		}
	}

	GIVEN("a 3x3x4 tensor filled with 3 and a 3x4 matrix filled with 2")
	{
		Matrix m(3, 4);
		m.FillWith(2);

		Tensor t({3, 3, 4});
		t.FillWith(3);

		WHEN("multiplying them together")
		{
			Tensor res = t * m;
			THEN("the result is a 3x4x4 tensor")
			{
				REQUIRE(res.GetShape().size() == 3);
				REQUIRE(res.GetShape()[0] == 3);
				REQUIRE(res.GetShape()[1] == 4);
				REQUIRE(res.GetShape()[2] == 4);
			}
			THEN("every value in the new tensor is 18")
			{
				for (int i = 0; i < res.GetElementCount(); ++i)
					REQUIRE(res.GetValue(i) == 18);
			}
		}
		WHEN("multiplying the tensor with the matrix")
		{
			t *= m;
			THEN("the result is a 3x4x4 tensor")
			{
				REQUIRE(t.GetShape().size() == 3);
				REQUIRE(t.GetShape()[0] == 3);
				REQUIRE(t.GetShape()[1] == 4);
				REQUIRE(t.GetShape()[2] == 4);
			}
			THEN("every value in the new tensor is 18")
			{
				for (int i = 0; i < t.GetElementCount(); ++i)
					REQUIRE(t.GetValue(i) == 18);
			}
		}
	}
}

SCENARIO("using tensor-tensor arithmetics", "[tensor][math]")
{
	GIVEN("two 3x5x4x2 sized tensor, the first filled with 5, the second 1")
	{
		Tensor t1({3, 5, 4, 2});
		Tensor t2({3, 5, 4, 2});

		t1.FillWith(5);
		t2.FillWith(1);

		WHEN("adding them together")
		{
			Tensor res = t1 + t2;
			THEN("every value is 6")
			{
				for (unsigned int i = 0; i < res.GetElementCount(); i++)
					REQUIRE(res.GetValue(i) == 6);
			}
		}
		WHEN("substracting them from each other")
		{
			Tensor res = t1 - t2;
			THEN("every value is 4")
			{
				for (unsigned int i = 0; i < res.GetElementCount(); i++)
					REQUIRE(res.GetValue(i) == 4);
			}
		}
		WHEN("adding the second one to the first")
		{
			t1 += t2;
			THEN("every value is 6")
			{
				for (unsigned int i = 0; i < t1.GetElementCount(); i++)
					REQUIRE(t1.GetValue(i) == 6);
			}
		}
		WHEN("subtracting the second one from the first")
		{
			t1 -= t2;
			THEN("every value is 6")
			{
				for (unsigned int i = 0; i < t1.GetElementCount(); i++)
					REQUIRE(t1.GetValue(i) == 4);
			}
		}
	}

	GIVEN("two tensors: 3x2x4x4 filled with 2, 2x5x4x4 filled with 5")
	{
		Tensor t1({3, 2, 2, 4});
		Tensor t2({2, 5, 2, 4});

		t1.FillWith(2);
		t2.FillWith(5);

		WHEN("multiplying them together")
		{
			Tensor res = t1 * t2;
			THEN("the resulting tensor is 3x5x2x4")
			{
				REQUIRE(res.GetShape().size() == 4);
				REQUIRE(res.GetShape()[0] == 3);
				REQUIRE(res.GetShape()[1] == 5);
			}
			THEN("every value is 20")
			{
				for (int i = 0; i < res.GetElementCount(); ++i)
					REQUIRE(res.GetValue(i) == 20);
			}
		}
		WHEN("multiplying the first with the second")
		{
			t1 *= t2;

			THEN("the resulting tensor is 3x5x2x4")
			{
				REQUIRE(t1.GetShape().size() == 4);
				REQUIRE(t1.GetShape()[0] == 3);
				REQUIRE(t1.GetShape()[1] == 5);
			}
			THEN("every value is 20")
			{
				for (int i = 0; i < t1.GetElementCount(); ++i)
					REQUIRE(t1.GetValue(i) == 20);
			}
		}
	}
}