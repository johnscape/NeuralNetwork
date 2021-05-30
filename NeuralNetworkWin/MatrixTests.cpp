#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "Matrix.h"
#include "MatrixException.hpp"

SCENARIO("Using an empty matrix", "[matrix]")
{
	GIVEN("an empty matrix")
	{
		Matrix mat;
		WHEN("getting the matrix's properties")
		{
			THEN("the row and column number is 1")
			{
				REQUIRE(mat.GetColumnCount() == 1);
				REQUIRE(mat.GetRowCount() == 1);
			}
			THEN("the 0th element is 0")
			{
				REQUIRE(mat[0] == 0);
			}
			THEN("the 1st element cannot be accessed")
			{
				REQUIRE_THROWS_AS(mat[1], MatrixIndexException);
			}
		}
		WHEN("changing the first item to 1")
		{
			mat.SetValue(0, 1);
			THEN("the first value is 1")
			{
				REQUIRE(mat[0] == 1);
			}
		}
	}
}

SCENARIO("Using a 3x3 matrix", "[matrix]")
{
	GIVEN("a set of number 0..3")
	{
		float vals[4] = { 0, 1, 2, 3 };
		WHEN("creating a 2x2 matrix with the set")
		{
			Matrix mat(2, 2, vals);
			THEN("the matix equals with the set")
			{
				REQUIRE(mat[0] == 0);
				REQUIRE(mat[1] == 1);
				REQUIRE(mat[2] == 2);
				REQUIRE(mat[3] == 3);
			}
		}
	}
	GIVEN("a 3x3 matrix without initial values")
	{
		Matrix mat(3, 3);
		WHEN("no command given")
		{
			THEN("the row and column number is 3")
			{
				REQUIRE(mat.GetColumnCount() == 3);
				REQUIRE(mat.GetRowCount() == 3);
			}
			THEN("the count of elements is 9")
			{
				REQUIRE(mat.GetElementCount() == 9);
			}
			THEN("the first and final value is 0")
			{
				REQUIRE(mat[0] == 0);
				REQUIRE(mat[8] == 0);
			}
		}
		WHEN("setting the 4th value to 3")
		{
			mat.SetValue(4, 3);
			THEN("the 4th value is 3")
			{
				REQUIRE(mat[4] == 3);
			}
		}
	}
	GIVEN("a 3x3 matrix with initial values")
	{
		float vals[9] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
		Matrix mat(3, 3, vals);
		WHEN("getting the values from the matrix")
		{
			THEN("the two value getter function is equal")
			{
				REQUIRE(mat[5] == mat.GetValue(5));
			}
			THEN("both of the overloads of the GetValue is the same")
			{
				REQUIRE(mat.GetValue(3) == mat.GetValue(1, 0));
			}
		}
		WHEN("adjusting two value")
		{
			mat.AdjustValue(0, 2);
			mat.AdjustValue(1, 1, 5);
			THEN("the first value is 3 and the (1, 1) is 5+5")
			{
				REQUIRE(mat[0] == 3);
				REQUIRE(mat.GetValue(1, 1) == 10);
			}
		}
		WHEN("getting the vector size")
		{
			THEN("returns an exception")
			{
				REQUIRE_THROWS_AS(mat.GetVectorSize(), MatrixVectorException);
			}
		}
		WHEN("reseting the matrix")
		{
			mat.Reset(1, 2);
			THEN("all values is 0")
			{
				REQUIRE(mat[0] == 0);
				REQUIRE(mat[1] == 0);
			}
			THEN("the row number is 1")
			{
				REQUIRE(mat.GetRowCount() == 1);
			}
			THEN("the column number is 2")
			{
				REQUIRE(mat.GetColumnCount() == 2);
			}
		}
	}
}

SCENARIO("Using a 3x1 vector", "[matrix]")
{
	GIVEN("a 3x1 vector with initializing values")
	{
		float vals[3] = { 1, 2, 3 };
		Matrix vec(3, 1, vals);
		WHEN("getting the vector's size")
		{
			THEN("the row number is 3")
			{
				REQUIRE(vec.GetRowCount() == 3);
			}
			THEN("the column number is 1")
			{
				REQUIRE(vec.GetColumnCount() == 1);
			}
		}
		WHEN("getting the vector's size")
		{
			THEN("the answer is 3 as the number of rows")
			{
				REQUIRE(vec.GetVectorSize() == vec.GetRowCount());
			}
		}
	}
}

SCENARIO("Loading a matrix from the other", "[matrix]")
{
	GIVEN("an empty matrix and a 2x2 initialized matrix")
	{
		Matrix empty;
		
		float vals[9] = { 1, 2, 3, 4 };
		Matrix mat(2, 2, vals);

		WHEN("loading the empty from the initialized")
		{
			empty.ReloadFromOther(mat);
			THEN("the row number of the empty is equal to the initialized")
			{
				REQUIRE(mat.GetRowCount() == empty.GetRowCount());
			}
			THEN("the column number of the empty is equal to the initialized")
			{
				REQUIRE(mat.GetColumnCount() == empty.GetColumnCount());
			}
			THEN("the values in the matrices are be equal")
			{
				REQUIRE(mat[0] == empty[0]);
				REQUIRE(mat[1] == empty[1]);
				REQUIRE(mat[2] == empty[2]);
				REQUIRE(mat[3] == empty[3]);
			}
		}
	}
	GIVEN("a 2x2 initialized matrix")
	{
		float vals[9] = { 1, 2, 3, 4 };
		Matrix mat(2, 2, vals);
		WHEN("constructing the matrix from this")
		{
			Matrix mat2(mat);
			THEN("the row number of the new is equal to the initialized")
			{
				REQUIRE(mat.GetRowCount() == mat2.GetRowCount());
			}
			THEN("the column number of the new is equal to the initialized")
			{
				REQUIRE(mat.GetColumnCount() == mat2.GetColumnCount());
			}
			THEN("the values in the matrices are be equal")
			{
				REQUIRE(mat[0] == mat2[0]);
				REQUIRE(mat[1] == mat2[1]);
				REQUIRE(mat[2] == mat2[2]);
				REQUIRE(mat[3] == mat2[3]);
			}
		}
	}
}

SCENARIO("using matrix operators", "[matrix]")
{
	GIVEN("two different matrix, a and b")
	{
		Matrix a(5, 3);
		Matrix b(8, 1);
		
		WHEN("assigning b to a")
		{
			a = b;
			THEN("a is equals to b")
			{
				REQUIRE(a == b);
			}
		}
	}
}

SCENARIO("creating a submatrix", "[matrix]")
{
	GIVEN("A 3x3 initialized matrix")
	{
		float vals[9] = { 0,1,2,3,4,5,6,7,8 };
		Matrix m(3, 3, vals);
		WHEN("Creating a 2x2 submatrix")
		{
			Matrix sub = m.GetSubMatrix(0, 0, 2, 2);
			THEN("The new matrix is 2x2")
			{
				REQUIRE(sub.GetColumnCount() == 2);
				REQUIRE(sub.GetRowCount() == 2);
			}
			THEN("The matrix has 0, 1, 3, 4 as values")
			{
				REQUIRE(sub[0] == 0);
				REQUIRE(sub[1] == 1);
				REQUIRE(sub[2] == 3);
				REQUIRE(sub[3] == 4);
			}
		}
	}
}