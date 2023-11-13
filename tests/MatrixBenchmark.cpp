#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include "../include/NeuralNetwork/Matrix.h"
#include "../include/NeuralNetwork/MatrixException.hpp"

TEST_CASE("Matrix benchmarks")
{
    Matrix small(128, 128);
    Matrix medium(1024, 1024);
    Matrix large(4096, 4096);

    small.FillWithRandom();
    medium.FillWithRandom();
    large.FillWithRandom();

    /*BENCHMARK("Matrix addition small")
    {
        return small + small;
    };

    BENCHMARK("Matrix multiplication small")
    {
        return small * small;
    };

    BENCHMARK("Matrix addition medium")
    {
        return medium + medium;
    };

    BENCHMARK("Matrix multiplication medium")
    {
        return medium * medium;
    };

    BENCHMARK("Matrix addition large")
    {
        return large + large;
    };

    BENCHMARK("Matrix multiplication large")
    {
        return large * large;
    };*/
    BENCHMARK("Matrix multiplication - small naive")
    {
        return small.SlowMultiply(small);
    };

    BENCHMARK("Matrix multiplication - medium naive")
    {
        return medium.SlowMultiply(medium);
    };

    BENCHMARK("Matrix multiplication - large naive")
    {
        return large.SlowMultiply(large);
    };

    BENCHMARK("Matrix multiplication - small SIMD")
    {
        return small * small;
    };

    BENCHMARK("Matrix multiplication - medium SIMD")
    {
        return medium * medium;
    };

    BENCHMARK("Matrix multiplication - large SIMD")
    {
        return large * large;
    };
}