#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "Matrix.h"
#include "MatrixException.hpp"

SCENARIO("matrix initialization", "[matrix]")
{
	GIVEN("two number 3 and 4")
	{
		unsigned int a = 3;
		unsigned int b = 4;

		WHEN("initializing a matrix with these values")
		{
			Matrix m(a, b);
			THEN("the row cound is 3 and the column count is 4")
			{
				REQUIRE(m.GetRowCount() == 3);
				REQUIRE(m.GetColumnCount() == 4);
			}
		}
	}
	GIVEN("two number 2 and 4 and an array of floats")
	{
		unsigned int a = 2;
		unsigned int b = 4;
		float values[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };

		WHEN("initializing a matrix with these values")
		{
			Matrix m(a, b, values);
			THEN("the row cound is 2 and the column count is 4")
			{
				REQUIRE(m.GetRowCount() == 2);
				REQUIRE(m.GetColumnCount() == 4);
			}
			THEN("the matrix's values are the same as in the array")
			{
				for (unsigned int i = 0; i < 8; i++)
					REQUIRE(m.GetValue(i) == values[i]);
			}
		}
	}
	GIVEN("an empty matrix")
	{
		Matrix m;
		WHEN("getting the properties of the matrix")
		{
			THEN("the row and column number is 1")
			{
				REQUIRE(m.GetColumnCount() == 1);
				REQUIRE(m.GetRowCount() == 1);
			}
			THEN("the only value is 0")
			{
				REQUIRE(m[0] == 0);
			}
		}
	}
	GIVEN("a 3x2 matrix and a 0 matrix")
	{
		Matrix a;
		Matrix b(3, 2);

		WHEN("assigning the second matrix to the first")
		{
			a = b;
			THEN("the first is 3x2")
			{
				REQUIRE(a.GetRowCount() == 3);
				REQUIRE(a.GetColumnCount() == 2);
			}
		}
	}
	GIVEN("two 3x3 matrix, one empty and one randomly initialized")
	{
		Matrix a(3, 3);
		Matrix b(3, 3);

		b.FillWithRandom();

		WHEN("copying b to a")
		{
			a.Copy(b);

			THEN("the two matrix are equal")
			{
				REQUIRE(a == b);
			}
		}
	}
	GIVEN("an unsigned int value of 5")
	{
		unsigned int i = 5;
		WHEN("creating an eye matrix")
		{
			Matrix eye = Matrix::Eye(i);
			THEN("the matrix is 5x5")
			{
				REQUIRE(eye.GetRowCount() == 5);
				REQUIRE(eye.GetColumnCount() == 5);
			}
			THEN("the values in the main diagonal are 1")
			{
				for (unsigned int i = 0; i < 5; i++)
					REQUIRE(eye.GetValue(i, i) == 1);
			}
			THEN("the other values are 0")
			{
				for (unsigned int r = 0; r < 5; r++)
					for (unsigned int c = 0; c < 5; c++)
						if (r != c)
							REQUIRE(eye.GetValue(r, c) == 0);
			}
		}
	}
	//TODO: Test concat
}

SCENARIO("setting and getting matrix values", "[matrix]")
{
	GIVEN("a 2x2 initialized matrix")
	{
		float vals[4] = { 0, 1, 2, 3 };
		Matrix mat(2, 2, vals);

		WHEN("getting the values of the matrix by index")
		{
			THEN("the values are 0, 1, 2, 3")
			{
				for (unsigned int i = 0; i < 4; i++)
					REQUIRE(mat[i] == i);
			}
			THEN("the output of the two function ([] and GetValue) is the same")
			{
				for (unsigned int i = 0; i < 4; i++)
					REQUIRE(mat[i] == mat.GetValue(i));
			}
		}
		WHEN("getting the value by row and column number")
		{
			float val = mat.GetValue(1, 1);
			THEN("the value is 3")
			{
				REQUIRE(val == 3);
			}
		}
	}
	GIVEN("a 2x2 empty matrix")
	{
		Matrix mat(2, 2);
		WHEN("setting the value by SetValue")
		{
			for (unsigned int i = 0; i < 4; i++)
				mat.SetValue(i, i);
			THEN("the values are 0, 1, 2, 3")
			{
				for (unsigned int i = 0; i < 4; i++)
					REQUIRE(mat[i] == i);
			}
		}
		WHEN("setting the value by row and col number")
		{
			for (unsigned int r = 0; r < 2; r++)
				for (unsigned int c = 0; c < 2; c++)
					mat.SetValue(r, c, r);
			THEN("the row number is in every row")
			{
				for (unsigned int r = 0; r < 2; r++)
					for (unsigned int c = 0; c < 2; c++)
						REQUIRE(mat.GetValue(r, c) == r);
			}
		}
		WHEN("adjusting values")
		{
			mat.AdjustValue(0, 2);
			mat.AdjustValue(0, 2);

			mat.AdjustValue(1, 1, 1);
			mat.AdjustValue(1, 1, 1);

			THEN("the first value is 4 and the last is 2")
			{
				REQUIRE(mat.GetValue(0) == 4);
				REQUIRE(mat.GetValue(3) == 2);
			}
		}
		WHEN("filling the matrix with a constant value")
		{
			mat.FillWith(5);
			THEN("every value is 5")
			{
				for (unsigned int i = 0; i < 4; i++)
					REQUIRE(mat[i] == 5);
			}
		}
		WHEN("filling the matrix with random values")
		{
			mat.FillWithRandom();
			THEN("every value is between -1 and 1")
			{
				for (unsigned int i = 0; i < 4; i++)
				{
					REQUIRE(mat[i] <= 1);
					REQUIRE(mat[i] >= -1);
				}
			}
		}
	}
	GIVEN("a 3x4 matrix with values of 0..11")
	{
		Matrix mat(3, 4);
		for (size_t i = 0; i < 12; i++)
			mat.SetValue(i, i);

		WHEN("transposing the matrix")
		{
			mat.Transpose();

			THEN("the second value is 4")
			{
				REQUIRE(mat[1] == 4);
			}
		}
	}
}

SCENARIO("using vector operations", "[matrix][vector]")
{
	GIVEN("a 3x1 empty vector")
	{
		Matrix vec(3, 1);
		WHEN("getting the vector's size")
		{
			THEN("the result is 3")
			{
				REQUIRE(vec.GetVectorSize() == 3);
			}
		}
	}
	GIVEN("a 2x3 empty matrix")
	{
		Matrix mat(2, 3);
		WHEN("getting the vector's size")
		{
			THEN("throws MatrixVectorException")
			{
				REQUIRE_THROWS_AS(mat.GetVectorSize(), MatrixVectorException);
			}
		}
	}
	GIVEN("two vectors, a 3x1 and a 1x4, filled with two")
	{
		Matrix a(3, 1);
		Matrix b(1, 4);

		a.FillWith(2);
		b.FillWith(4);

		WHEN("getting their outer product")
		{
			Matrix outer = a.OuterProduct(b);

			THEN("the result is 3x4")
			{
				REQUIRE(outer.GetRowCount() == 3);
				REQUIRE(outer.GetColumnCount() == 4);
			}
			THEN("all values are 8")
			{
				for (unsigned int i = 0; i < outer.GetElementCount(); i++)
					REQUIRE(outer[i] == 8);
			}
		}
	}
	GIVEN("two vectors, a 3x1 and a 1x3, filled with two")
	{
		Matrix a(3, 1);
		Matrix b(1, 3);

		a.FillWith(2);
		b.FillWith(4);

		WHEN("getting their dot product")
		{
			float dot = a.DotProcudt(b);

			THEN("the result is 24")
			{
				REQUIRE(dot == 24);
			}
		}
	}
}

SCENARIO("reseting matrix", "[matrix]")
{
	GIVEN("a 2x3 randomly initialized matrix and a 1x2 zero matrix")
	{
		Matrix a(2, 3);
		a.FillWithRandom();
		Matrix b(1, 2);
		WHEN("reseting the first matrix to 3x3")
		{
			a.Reset(3, 3);
			THEN("the row and column numbers are 3")
			{
				REQUIRE(a.GetRowCount() == 3);
				REQUIRE(a.GetColumnCount() == 3);
			}
			THEN("every value is 0")
			{
				for (unsigned int i = 0; i < 9; i++)
					REQUIRE(a[i] == 0);
			}
		}
		WHEN("reloading by from a")
		{
			b.ReloadFromOther(a);
			THEN("a is still randomly initialized")
			{
				REQUIRE(a[0] != 0);
			}
			THEN("a and b are equal")
			{
				REQUIRE(a == b);
			}
		}
	}

	//TODO: Test Transpose
}

SCENARIO("using matrix arithmetics", "[matrix][math]")
{
	GIVEN("two 2x2 matrices, filled with 1")
	{
		Matrix a(2, 2);
		Matrix b(2, 2);

		a.FillWith(1);
		b.FillWith(1);

		WHEN("adding them together")
		{
			Matrix c = a + b;

			THEN("the new matrix is filled with 2s")
			{
				for (unsigned int i = 0; i < 2; i++)
					REQUIRE(c[i] == 2);
			}
		}
		WHEN("adding b to a")
		{
			a += b;
			THEN("the a matrix is filled with 2s")
			{
				for (unsigned int i = 0; i < 2; i++)
					REQUIRE(a[i] == 2);
			}
		}
		WHEN("substracting b from a")
		{
			Matrix c = a - b;

			THEN("the new matrix is filled with 0s")
			{
				for (unsigned int i = 0; i < 2; i++)
					REQUIRE(c[i] == 0);
			}
		}
		WHEN("substracting b from a (-=)")
		{
			a -= b;

			THEN("the a matrix is filled with 0s")
			{
				for (unsigned int i = 0; i < 2; i++)
					REQUIRE(a[i] == 0);
			}
		}
	}
	GIVEN("a 3x3 matrix filled with 5 and a float of 2")
	{
		Matrix a(3, 3);
		a.FillWith(5);

		float b = 2;

		WHEN("multiplying them together")
		{
			Matrix c = a * b;
			THEN("all values in the new matrix are 10")
			{
				for (unsigned int i = 0; i < 9; i++)
					REQUIRE(c[i] == 10);
			}
		}
		WHEN("multiplying the matrix")
		{
			a *= b;

			THEN("all values are 10")
			{
				for (unsigned int i = 0; i < 9; i++)
					REQUIRE(a[i] == 10);
			}
		}
	}
	GIVEN("a 3x2 and a 2x4 matrix, first filled with ones, second filled with fives")
	{
		Matrix a(3, 2);
		Matrix b(2, 4);

		a.FillWith(1);
		b.FillWith(5);

		WHEN("multiplying them together")
		{
			Matrix c = a * b;

			THEN("the new matrix is 3x4")
			{
				REQUIRE(c.GetRowCount() == 3);
				REQUIRE(c.GetColumnCount() == 4);
			}
			THEN("the new matrix is filled with 10")
			{
				for (unsigned int i = 0; i < 12; i++)
					REQUIRE(c[i] == 10);
			}
		}
		WHEN("multiplying a with b")
		{
			a *= b;

			THEN("the a matrix is 3x4")
			{
				REQUIRE(a.GetRowCount() == 3);
				REQUIRE(a.GetColumnCount() == 4);
			}
			THEN("the a matrix is filled with 10")
			{
				for (unsigned int i = 0; i < 12; i++)
					REQUIRE(a[i] == 10);
			}
		}
	}
	GIVEN("two 3x3 matrices, one filled with 3, the other filled with 2")
	{
		Matrix a(3, 3);
		Matrix b(3, 3);

		a.FillWith(3);
		b.FillWith(2);

		WHEN("creating a new matrix with elementwise multiplaction")
		{
			Matrix c = Matrix::ElementwiseMultiply(a, b);

			THEN("the new matrix is 3x3")
			{
				REQUIRE(c.GetColumnCount() == 3);
				REQUIRE(c.GetRowCount() == 3);
			}
			THEN("every value in the new matrix is 6")
			{
				for (unsigned int i = 0; i < 9; i++)
					REQUIRE(c[i] == 6);
			}
		}
		WHEN("multiplying a with b, elementwise")
		{
			a.ElementwiseMultiply(b);

			THEN("the matrix is still 3x3")
			{
				REQUIRE(a.GetColumnCount() == 3);
				REQUIRE(a.GetRowCount() == 3);
			}
			THEN("every value in the matrix is 6")
			{
				for (unsigned int i = 0; i < 9; i++)
					REQUIRE(a[i] == 6);
			}
		}
	}
	GIVEN("a 5x5 matrix filled with 1")
	{
		Matrix a(5, 5);
		a.FillWith(1);

		WHEN("summing the matrix")
		{
			float sum = a.Sum();

			THEN("the result is 25")
			{
				REQUIRE(sum == 25);
			}
		}
	}
}

SCENARIO("getting sub-matrices", "[matrix]")
{
	GIVEN("a 5x5 randomly initialized matrix")
	{
		Matrix mat(5, 5);
		mat.FillWithRandom();

		WHEN("getting a 3x3 sub-matrix from (1, 1)")
		{
			Matrix sub = mat.GetSubMatrix(1, 1, 3, 3);

			THEN("the sub-matrix has 3 columns and rows")
			{
				REQUIRE(sub.GetRowCount() == 3);
				REQUIRE(sub.GetColumnCount() == 3);
			}
			THEN("the sub-matrix has the same values as the original")
			{
				REQUIRE(sub[0] == mat.GetValue(1, 1));
			}
		}
		WHEN("getting a row matrix from the first row")
		{
			Matrix row = mat.GetRowMatrix(0);
			THEN("the values in the row matrix are the same as in the original")
			{
				for (unsigned int i = 0; i < 5; i++)
					REQUIRE(row[i] == mat[i]);
			}
		}
		WHEN("getting a column matrix from the second column")
		{
			Matrix col = mat.GetColumnMatrix(1);
			THEN("the values in the column matrix are the same as in the original")
			{
				for (unsigned int i = 0; i < 5; i++)
					REQUIRE(col[i] == mat.GetValue(i, 1));
			}
		}
	}

}

SCENARIO("using matrix comparation", "[matrix]")
{
	GIVEN("two, equal matrix")
	{
		Matrix a(3, 3);
		Matrix b(3, 3);

		WHEN("checkig equality")
		{
			THEN("the result is true")
			{
				REQUIRE(a == b);
				REQUIRE_FALSE(a != b);
			}
		}
		WHEN("checking size")
		{
			THEN("the two has the same size")
			{
				REQUIRE(a.IsSameSize(b));
			}
		}
	}
}