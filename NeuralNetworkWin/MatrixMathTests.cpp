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
			THEN("the result is the same as using AddIn function")
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
}