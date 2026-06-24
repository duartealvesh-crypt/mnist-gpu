#include "matrix.h"
#include <cstdio>

// Minimal sanity check for the Matrix GPU kernels: build a constant matrix,
// transpose it on the device, copy the result back and print it.
#define N 9
#define M 4

int main()
{
    Matrix z1(N, M);
    for (unsigned r = 0; r < N; ++r)
        for (unsigned c = 0; c < M; ++c)
            z1(r, c) = 1.0;
    z1.CPU_to_GPU();

    // Any number of GPU operations can be chained from here.
    Matrix z3 = z1.transpose(); // (M, N)

    // Bring the result back to host memory once computation is done.
    z3.GPU_to_CPU();
    z3.print(false);

    return 0;
}
