#include "matrix.h"
#include "err.h"
#include <stdlib.h>
#include <stdio.h>

#define N 9
#define M 4
#define Z 3


int main(int argc, char *argv[])
{
    
    // Allocate matching CPU and GPU storage for each matrix.
    matrix_t * z1;
    matrix_t *z2;
    matrix_t *z3;
    z1 = alloc_matrix(N, M);
    z2 = alloc_matrix(N, N);
    z3 = alloc_matrix(M,N);

    // Initialize matrices on the host.
    for (int idx = 0; idx < z1->columns * z1->rows; idx++) {
        z1->m[idx] = 1.0;
    }
    for (int idx = 0; idx < z2->columns * z2->rows; idx++) {
        z2->m[idx] = 0.0;
    }

    // Push the initial values to the GPU.
    matrix_CPU_to_GPU(z1);
    matrix_CPU_to_GPU(z2);

    // Any number of GPU operations can be chained from here.
    matrix_transpose_gpu(z1,z3);

    // Bring the result back to host memory once computation is done.
    matrix_GPU_to_CPU(z3);

    // Print the result to sanity-check the operation.
    print_matrix(z3, false);

    destroy_matrix(z1);
    destroy_matrix(z2);
    destroy_matrix(z3);




    return 0;
}
