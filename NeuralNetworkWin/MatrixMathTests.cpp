#include "catch.hpp"
#include "Matrix.h"
#include "MatrixMath.h"

SCENARIO("Using matrix operations", "[matrix]")
{
	GIVEN("two 3x3 matrices, a and b, and a 2x2 matrix, c")
	{
		Matrix a, b, c;
		float vals[9] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
		Matrix temp(3, 3, vals);
		c.Reset(2, 2);
		
		WHEN("checking the similarities of the matrices")
		{
			a.ReloadFromOther(temp);
			b.ReloadFromOther(temp);

			THEN("the size of a and b is equal")
			{
				REQUIRE(MatrixMath::SizeCheck(a, b));
			}
			THEN("the size of a and c in not equal")
			{
				REQUIRE_FALSE(MatrixMath::SizeCheck(a, c));
			}
			THEN("a and b is equal")
			{
				REQUIRE(MatrixMath::IsEqual(a, b));
				REQUIRE(a == b);
			}
			THEN("a and c is not equal")
			{
				REQUIRE_FALSE(MatrixMath::IsEqual(a, c));
				REQUIRE_FALSE(a == c);
				REQUIRE(a != c);
			}
		}
		WHEN("filling up matrix c with the value 3")
		{
			MatrixMath::FillWith(c, 3);
			THEN("the first and final value is 3")
			{
				REQUIRE(c[0] == 3);
				REQUIRE(c[3] == 3);
			}
		}
		WHEN("filling up c and summing the values of c")
		{
			MatrixMath::FillWith(c, 1);
			float sum = MatrixMath::Sum(c);
			THEN("the sum of the values is 4")
			{
				REQUIRE(sum == 4);
			}
		}
		WHEN("using random filling")
		{
			MatrixMath::FillWithRandom(c, 0, 1);
			THEN("the sum of c must be larger than 0")
			{
				REQUIRE(MatrixMath::Sum(c) > 0);
			}
		}
		WHEN("copying matrix a to b")
		{
			MatrixMath::FillWithRandom(a);
			MatrixMath::Copy(a, b);
			THEN("a is equals to b")
			{
				REQUIRE(a == b);
			}
		}
	}
	GIVEN("two 2x2 initialized matrix")
	{
		float vals1[4] = { 0, 1, 2, 3 };
		float vals2[4] = { 4, 5, 6, 7 };

		Matrix a(2, 2, vals1);
		Matrix b(2, 2, vals2);

		WHEN("concatenating the matrices by the rows")
		{
			Matrix result = MatrixMath::Concat(a, b, 0);
			THEN("the result has 4 rows")
			{
				REQUIRE(result.GetRowCount() == 4);
			}
			THEN("the result has 2 columns")
			{
				REQUIRE(result.GetColumnCount() == 2);
			}
			THEN("the third item is 2")
			{
				REQUIRE(result[2] == 2);
			}

		}
		WHEN("concatenating the matrices by the columns")
		{
			Matrix result = MatrixMath::Concat(a, b, 1);
			THEN("the result has 2 rows")
			{
				REQUIRE(result.GetRowCount() == 2);
			}
			THEN("the result has 4 columns")
			{
				REQUIRE(result.GetColumnCount() == 4);
			}
			THEN("the third item is 4")
			{
				REQUIRE(result[2] == 4);
			}
		}
	}
}

SCENARIO("Using matrix substraction", "[matrix][math]")
{
	GIVEN("matrix a and b, both 2x2")
	{
		float set1[4] = { 0, 1, 2, 3 };
		float set2[4] = { 4, 5, 6, 7 };

		Matrix a(2, 2, set1);
		Matrix b(2, 2, set2);

		WHEN("substracting a from b")
		{
			Matrix c = MatrixMath::Substract(b, a);
			THEN("the result is 4, 4, 4, 4")
			{
				REQUIRE(c[0] == 4);
				REQUIRE(c[1] == 4);
				REQUIRE(c[2] == 4);
				REQUIRE(c[3] == 4);
			}
		}
		WHEN("substracting b from a (-=)")
		{
			MatrixMath::SubstractIn(b, a);
			THEN("the result is 4, 4, 4, 4")
			{
				REQUIRE(b[0] == 4);
				REQUIRE(b[1] == 4);
				REQUIRE(b[2] == 4);
				REQUIRE(b[3] == 4);
			}
		}
		WHEN("using substraction operator overloading")
		{
			Matrix c = a - b;
			THEN("the result is the same as in previous tests")
			{
				Matrix c2 = MatrixMath::Substract(a, b);
				REQUIRE(c == c2);
			}
		}
		WHEN("using substraction and assignment operator")
		{
			Matrix tmp(a);
			a -= b;
			THEN("the result is the same as using SubstractIn function")
			{
				MatrixMath::SubstractIn(tmp, b);
				REQUIRE(a == tmp);
			}
		}
	}
	GIVEN("a 2x3 and a 4x3 randomly initialized matrix")
	{
		Matrix a(2, 3);
		Matrix b(4, 3);
		MatrixMath::FillWithRandom(a);
		MatrixMath::FillWithRandom(b);
		
		WHEN("adding the two matrices together")
		{
			THEN("an error appears")
			{
				REQUIRE_THROWS_AS(MatrixMath::Add(a, b), MatrixException);
			}
		}
	}

}

SCENARIO("Using matrix addition", "[matrix][math]")
{
	GIVEN("matrix a and b, both 2x2")
	{
		float set1[4] = { 0, 1, 2, 3 };
		float set2[4] = { 4, 5, 6, 7 };

		Matrix a(2, 2, set1);
		Matrix b(2, 2, set2);

		WHEN("adding the two matrix together")
		{
			Matrix c = MatrixMath::Add(a, b);
			THEN("the result is 4, 6, 8, 10")
			{
				REQUIRE(c[0] == 4);
				REQUIRE(c[1] == 6);
				REQUIRE(c[2] == 8);
				REQUIRE(c[3] == 10);
			}
		}
		WHEN("adding the second matrix to the first")
		{
			MatrixMath::AddIn(a, b);
			THEN("the result is 4, 6, 8, 10")
			{
				REQUIRE(a[0] == 4);
				REQUIRE(a[1] == 6);
				REQUIRE(a[2] == 8);
				REQUIRE(a[3] == 10);
			}
		}
		WHEN("using addition operator overloading")
		{
			Matrix c = a + b;
			THEN("the result is the same as in previous tests")
			{
				Matrix c2 = MatrixMath::Add(a, b);
				REQUIRE(c == c2);
			}
		}
		WHEN("using addition and assignment operator")
		{
			Matrix tmp(a);
			a += b;
			THEN("the result is the same as using AddIn function")
			{
				MatrixMath::AddIn(tmp, b);
				REQUIRE(a == tmp);
			}
		}
	}
	GIVEN("a 2x3 and a 4x3 randomly initialized matrix")
	{
		Matrix a(2, 3);
		Matrix b(4, 3);
		MatrixMath::FillWithRandom(a);
		MatrixMath::FillWithRandom(b);

		WHEN("substracting the matrices")
		{
			THEN("an error appears")
			{
				REQUIRE_THROWS_AS(MatrixMath::Substract(a, b), MatrixException);
			}
		}
	}
	GIVEN("a 2x2 zero matrix and a float value of 3")
	{
		float f = 3;
		Matrix mat(2, 2);
		WHEN("adding a float to a matrix")
		{
			MatrixMath::Add(mat, f);
			THEN("all the values are 3")
			{
				REQUIRE(mat[0] == 3);
				REQUIRE(mat[1] == 3);
				REQUIRE(mat[2] == 3);
				REQUIRE(mat[3] == 3);
			}
		}
	}
}

SCENARIO("Using matrix multiplications", "[matrix][math]")
{
	GIVEN("two matrices: 2x3 and 3x1")
	{
		float set1[6] = { 0, 1, 2, 3, 4, 5 };
		float set2[3] = { 1, 1, 1 };

		Matrix a(2, 3, set1);
		Matrix b(3, 1, set2);

		WHEN("multiplying the matrices with Multiply(a, b)")
		{
			Matrix c1 = MatrixMath::Multiply(a, b);
			THEN("the result is a 2x1 vector")
			{
				REQUIRE(c1.GetRowCount() == 2);
				REQUIRE(c1.GetColumnCount() == 1);
			}
			THEN("the result is 3 and 12")
			{
				REQUIRE(c1[0] == 3);
				REQUIRE(c1[1] == 12);
			}
		}
		WHEN("multiplying the matrices with Multiply(a, b, result)")
		{
			Matrix c2(2, 1);
			MatrixMath::Multiply(a, b, c2);
			THEN("the result is 3 and 12")
			{
				REQUIRE(c2[0] == 3);
				REQUIRE(c2[1] == 12);
			}
		}
		WHEN("multiplying the matrix with a constant float of 1")
		{
			Matrix res = MatrixMath::Multiply(a, 1);
			THEN("the result is the same as a")
			{
				REQUIRE(res == a);
			}
		}
		WHEN("multiplying the matrix with a float of 1 using operators")
		{
			Matrix res = a * 1;
			THEN("the result is the same as a")
			{
				REQUIRE(res == a);
			}
		}
		WHEN("using operator overloading")
		{
			Matrix c = a * b;
			THEN("the result is the same as using the function")
			{
				Matrix c2 = MatrixMath::Multiply(a, b);
				REQUIRE(c == c2);
			}
		}
		WHEN("using a float of 0 with MultiplyIn")
		{
			MatrixMath::MultiplyIn(a, 0);
			THEN("matrix a is zero")
			{
				REQUIRE(MatrixMath::Sum(a) == 0);
			}
		}
		WHEN("using a float of 0 with operator overloading and assignment")
		{
			a *= 0;
			THEN("matrix a is zero")
			{
				REQUIRE(MatrixMath::Sum(a) == 0);
			}
		}
	}
}