#ifndef MATRIX_H
#define MATRIX_H
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>

typedef struct
{
    double * m;
    unsigned columns;
    unsigned rows;
    double * m_gpu;
}  matrix_t;

// matrix.h
typedef enum { FUNC_ADDITION, FUNC_SIGMOID, FUNC_RELU, FUNC_DSIGMOID } func_id_t;

matrix_t * alloc_matrix(unsigned rows, unsigned columns);

void destroy_matrix(matrix_t *m);

void print_matrix(matrix_t *m, bool is_short);

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_CPU_to_GPU(matrix_t * m);

void matrix_GPU_to_CPU(matrix_t * m);

__global__
void matrix_dot_kernel(double *m1, double *m2, double *res, unsigned m1_rows,unsigned m1_columns, unsigned m2_columns);

__global__
void matrix_sum_kernel(double *m1, double *m2, double *res);


void matrix_sum_gpu(matrix_t *m1, matrix_t *m2, matrix_t *res);

__global__
void matrix_hadamard_kernel(double *m1, double *m2, double *res);

__global__
void matrix_minus_kernel(double *m1, double *m2, double *res);

void matrix_dot_gpu(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_hadamard_gpu(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_minus_gpu(matrix_t *m1, matrix_t *m2, matrix_t *res);

matrix_t * alloc_matrix_gpu(matrix_t * res, unsigned columns, unsigned rows);

void matrix_function(matrix_t *m1, double (*f)(double), matrix_t *res);

__global__
void matrix_function_kernel(double *m1, func_id_t f, double *res, int size);

void matrix_function_gpu(matrix_t *m1, func_id_t f, matrix_t *res);

void matrix_transpose_gpu(matrix_t *m1, matrix_t *res);

__global__
void matrix_transpose_kernel(double *m1, double *res, unsigned rows, unsigned cols);

void matrix_scalar_gpu(matrix_t *m1, double s, matrix_t *res);

__global__
void matrix_scalar_kernel(double *m1, double s, double *res);

void matrix_transpose(matrix_t *m1, matrix_t *res);

void matrix_scalar(matrix_t *m1, double s, matrix_t *res);

void matrix_memcpy(matrix_t *dest, const matrix_t *src);

#endif