#include "matrix.h"
#include <stdlib.h>
#include <string.h>
#include "err.h"
#define THREADS_PER_BLOCK 126
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

matrix_t * alloc_matrix(unsigned rows, unsigned columns)
{
    // CPU version
    matrix_t * res = (matrix_t*) malloc( sizeof(matrix_t) );
    res->m = (double *) calloc(columns * rows, sizeof(double));
    res->columns = columns;
    res->rows = rows;

    // GPU version : only for the matrix itself, (col and rows will be parameters)
    double* m_gpu;
    CHECK_ERROR(cudaMalloc((void **) &m_gpu, sizeof(double) * rows * columns));
    CHECK_ERROR(cudaMemcpy(m_gpu, res->m, sizeof(double) * rows * columns, cudaMemcpyHostToDevice));
    res->m_gpu = m_gpu;

    return res;
}

void matrix_CPU_to_GPU(matrix_t * m){
    CHECK_ERROR(cudaMemcpy(m->m_gpu, m->m, sizeof(double) * m->columns * m->rows, cudaMemcpyHostToDevice));
}

void matrix_GPU_to_CPU(matrix_t * m){
    CHECK_ERROR(cudaMemcpy(m->m, m->m_gpu, sizeof(double) * m->columns * m->rows, cudaMemcpyDeviceToHost));
}

void destroy_matrix(matrix_t *m)
{
    //printf("free %p %p\n", m, m->m);
    free(m->m);
    cudaFree(m->m_gpu);
    free(m);
}

void print_matrix(matrix_t *m, bool is_short){
    unsigned lim_rows = 0;
    unsigned lim_col = 0;

    if (is_short)
    {
        lim_rows = MIN(m->rows, 4);
        lim_col = MIN(m->columns, 10);
    }
    else
    {
        lim_rows = m->rows;
        lim_col = m->columns;
    }

    for (int row = 0; row < lim_rows; row ++)
    {
        for (int col = 0; col < lim_col; col ++)
        {
            printf("%.2lf ", m->m[col + row * m->columns]);
        }
        if (is_short && lim_col != m->columns) printf("...");
        printf("\n");
    }
    if (is_short && lim_rows != m->rows) printf("...\n");
}

__global__
void matrix_sum_kernel(double *m1, double *m2, double *res)
{

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    res[idx] = m1[idx] + m2[idx];

}

void matrix_sum_gpu(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));

    int blockDim = 32;
    int gridDim = (ceil(((float)m1->rows * m1->columns) / blockDim));
    matrix_sum_kernel<<<gridDim, blockDim>>>(m1->m_gpu, m2->m_gpu, res->m_gpu);

}


// --- CPU reference implementations -----------------------------------------
// matrix_sum, hadamard_product, matrix_minus, matrix_dot, matrix_function,
// matrix_transpose and matrix_scalar below are sequential, host-only versions.
// They are the original (pre-CUDA) baseline this project started from and are
// kept here for reference; the training pipeline in ann.cu only calls the
// *_gpu variants. See README.md for how this relates to running without a GPU.
void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    { 
        res->m[idx] = m1->m[idx] + m2->m[idx];
    }
}

void matrix_hadamard_gpu(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));

    int blockDim = 32;
    int gridDim = (ceil(((float)m1->rows * m1->columns) / blockDim));
    matrix_hadamard_kernel<<<gridDim, blockDim>>>(m1->m_gpu, m2->m_gpu, res->m_gpu);
    
   
}

__global__
void matrix_hadamard_kernel(double *m1, double *m2, double *res)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    res[idx] = m1[idx] * m2[idx];
}

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)   &&
             (m1->columns == res->columns)  &&
             (m1->rows == m2->rows)         &&
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
            res->m[idx] = m1->m[idx] * m2->m[idx];
    }
}


void matrix_minus_gpu(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));

    int blockDim = 32;
    int gridDim = (ceil(((float)m1->rows * m1->columns) / blockDim));
    matrix_minus_kernel<<<gridDim, blockDim>>>(m1->m_gpu, m2->m_gpu, res->m_gpu);
    
   
}

__global__
void matrix_minus_kernel(double *m1, double *m2, double *res)
{
    
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    res[idx] = m1[idx] - m2[idx];
    
}

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));
             
    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        res->m[idx] = m1->m[idx] - m2->m[idx];
    }
}


void matrix_dot_gpu(matrix_t *m1, matrix_t *m2, matrix_t *res){

    assert ( (m1->columns == m2->rows)  &&
             (m1->rows == res->rows)    &&
             (m2->columns == res->columns));

    dim3 blockDim(32, 32);
    dim3 gridDim(ceil(((float)m1->rows) / blockDim.x), ceil(((float)m2->columns) / blockDim.y));
    matrix_dot_kernel<<<gridDim, blockDim>>>(m1->m_gpu, m2->m_gpu, res->m_gpu, m1->rows, m1->columns, m2->columns);
    
}

__global__
void matrix_dot_kernel(double *m1, double *m2, double *res, unsigned m1_rows,unsigned m1_columns, unsigned m2_columns)
{
    

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x>=0 and x<m1_rows and y>=0 and y<m2_columns){
        double var = 0.0;

        for (int ii = 0; ii < m1_columns; ii++)
        {
            var += m1[ii + x * m1_columns] * m2[y + ii * m2_columns];
        }

        res[y + m2_columns * x] = var;
    }

    
}


void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->rows)  &&
             (m1->rows == res->rows)    &&
             (m2->columns == res->columns));

    for (int row = 0; row < m1->rows; row ++)
    {
        for (int col = 0; col < m2->columns; col ++)
        {
            int idx = col + row * m2->columns;
            double var = 0.0;

            for (int ii = 0; ii < m1->columns; ii++)
            {
                var += m1->m[ii + row * m1->columns] * m2->m[col + ii * m2->columns];
            }

            res->m[idx] = var;
        }
    }
}


void matrix_function(matrix_t *m1, double (*f)(double), matrix_t *res)
{
    assert ( (m1->columns == res->columns) &&             
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        res->m[idx] = f(m1->m[idx]);
    }
}

void matrix_function_gpu(matrix_t *m1, func_id_t f, matrix_t *res)
{
    // f must be a __device__ function (dispatched via apply_func below)
    assert ( (m1->columns == res->columns) &&
             (m1->rows == res->rows));

    int size = m1->rows * m1->columns;

    int blockDim = 32;
    int gridDim = (ceil(((float)m1->rows * m1->columns) / blockDim));
    matrix_function_kernel<<<gridDim, blockDim>>>(m1->m_gpu,f, res->m_gpu, size);
    
}

__device__ __host__ double apply_func(func_id_t f, double x) {
    switch(f) {
        case FUNC_ADDITION: return x * 2;
        case FUNC_SIGMOID:  return 1.0 / (1.0 + exp(-x));
        case FUNC_RELU:     return x > 0 ? x : 0;
        case FUNC_DSIGMOID: return 1.0 / (1.0 + exp(-x)) * (1-1.0 / (1.0 + exp(-x)));
        default: return x;
    }
}

__global__
void matrix_function_kernel(double *m1, func_id_t f, double *res, int size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx<size){
        res[idx] = apply_func(f, m1[idx]);
    }
    
}

void matrix_transpose(matrix_t *m1, matrix_t *res)
{
    assert ( (m1->columns == res->rows) &&             
             (m1->rows == res->columns));
    
    for (int row = 0; row < m1->rows; row++)
    {
        for (int col = 0; col < m1->columns; col ++)
        {
            res->m[row + col * m1->rows] = m1->m[col + row * m1->columns];
        }
    }
}

void matrix_transpose_gpu(matrix_t *m1, matrix_t *res)
{
    assert ( (m1->columns == res->rows) &&             
             (m1->rows == res->columns));



    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(((float)m1->rows) / blockDim.x), ceil(((float)m1->columns) / blockDim.y));
    matrix_transpose_kernel<<<gridDim, blockDim>>>(m1->m_gpu, res->m_gpu, m1->rows, m1->columns);

}

__global__
void matrix_transpose_kernel(double *m1, double *res, unsigned rows, unsigned cols)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    res[x + y * rows] = m1[y + x * cols]; // launches more threads than strictly needed, harmless

}



void matrix_scalar(matrix_t *m1, double s, matrix_t *res)
{
    assert ( (m1->rows == res->rows) &&             
             (m1->columns == res->columns));

    for (int idx = 0; idx < m1->columns*m1->rows; idx ++)
    {
        res->m[idx] = m1->m[idx] * s;
    }
}

void matrix_scalar_gpu(matrix_t *m1, double s, matrix_t *res)
{
    
    assert ( (m1->rows == res->rows) &&             
             (m1->columns == res->columns));

    int blockDim = 32;
    int gridDim = (ceil(((float)m1->rows * m1->columns) / blockDim));
    matrix_scalar_kernel<<<gridDim, blockDim>>>(m1->m_gpu, s, res->m_gpu);
}

__global__
void matrix_scalar_kernel(double *m1, double s, double *res)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    res[idx] = m1[idx] * s;
}

void matrix_memcpy(matrix_t *dest, const matrix_t *src)
{
    assert ( (dest->rows == src->rows)      &&             
             (dest->columns == src->columns));

    memcpy(dest->m, src->m, src->columns * src->rows * sizeof(double));
    cudaMemcpy(dest->m_gpu, src->m_gpu, src->columns * src->rows * sizeof(double), cudaMemcpyDeviceToDevice);     
}