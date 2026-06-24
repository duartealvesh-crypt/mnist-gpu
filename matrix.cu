#include "matrix.h"
#include <iostream>
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define THREADS_PER_BLOCK 128 // Aligned to Warp boundary (multiples of 32)

// ============================================================================
// --- CUDA KERNELS ---
// ============================================================================

__global__ void matrix_sum_kernel(double *m1, double *m2, double *res, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) res[idx] = m1[idx] + m2[idx];
}

__global__ void matrix_minus_kernel(double *m1, double *m2, double *res, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) res[idx] = m1[idx] - m2[idx];
}

__global__ void matrix_hadamard_kernel(double *m1, double *m2, double *res, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) res[idx] = m1[idx] * m2[idx];
}

__global__ void matrix_scalar_kernel(double *m1, double s, double *res, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) res[idx] = m1[idx] * s;
}

__global__ void matrix_dot_kernel(double *m1, double *m2, double *res, unsigned m1_rows, unsigned m1_cols, unsigned m2_cols) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < m1_rows && y < m2_cols) {
        double var = 0.0;
        for (int ii = 0; ii < m1_cols; ii++) {
            var += m1[ii + x * m1_cols] * m2[y + ii * m2_cols];
        }
        res[y + m2_cols * x] = var;
    }
}

__device__ double apply_func_device(func_id_t f, double x) {
    switch(f) {
        case FUNC_ADDITION: return x * 2;
        case FUNC_SIGMOID:  return 1.0 / (1.0 + exp(-x));
        case FUNC_RELU:     return x > 0 ? x : 0;
        case FUNC_DSIGMOID: {
            double sig = 1.0 / (1.0 + exp(-x));
            return sig * (1.0 - sig);
        }
        default: return x;
    }
}

__global__ void matrix_function_kernel(double *m1, func_id_t f, double *res, int size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) res[idx] = apply_func_device(f, m1[idx]);
}

__global__ void matrix_transpose_kernel(double *m1, double *res, unsigned rows, unsigned cols) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x < rows && y < cols) {
        res[x + y * rows] = m1[y + x * cols];
    }
}

// ============================================================================
// --- MATRIX CLASS MEMBER IMPLEMENTATIONS ---
// ============================================================================

Matrix::Matrix(unsigned r, unsigned c) : rows(r), columns(c), h_m(nullptr), d_m_gpu(nullptr) {
    h_m = (double*)calloc(rows * columns, sizeof(double));
    if (!h_m) throw std::runtime_error("Host memory allocation failed");

    if (cudaMalloc(&d_m_gpu, sizeof(double) * rows * columns) != cudaSuccess) {
        free(h_m);
        throw std::runtime_error("Device memory allocation failed");
    }
    cudaMemcpy(d_m_gpu, h_m, sizeof(double) * rows * columns, cudaMemcpyHostToDevice);
}

Matrix::~Matrix() {
    if (h_m) free(h_m);
    if (d_m_gpu) cudaFree(d_m_gpu);
}

// Move Constructor
Matrix::Matrix(Matrix&& other) noexcept 
    : rows(other.rows), columns(other.columns), h_m(other.h_m), d_m_gpu(other.d_m_gpu) {
    other.h_m = nullptr;
    other.d_m_gpu = nullptr;
    other.rows = 0;
    other.columns = 0;
}

// Move Assignment Operator
Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        if (h_m) free(h_m);
        if (d_m_gpu) cudaFree(d_m_gpu);

        rows = other.rows;
        columns = other.columns;
        h_m = other.h_m;
        d_m_gpu = other.d_m_gpu;

        other.h_m = nullptr;
        other.d_m_gpu = nullptr;
        other.rows = 0;
        other.columns = 0;
    }
    return *this;
}

void Matrix::CPU_to_GPU() {
    cudaMemcpy(d_m_gpu, h_m, sizeof(double) * rows * columns, cudaMemcpyHostToDevice);
}

void Matrix::GPU_to_CPU() {
    cudaMemcpy(h_m, d_m_gpu, sizeof(double) * rows * columns, cudaMemcpyDeviceToHost);
}

void Matrix::copy_from(const Matrix& src) {
    assert(rows == src.rows && columns == src.columns);
    memcpy(h_m, src.h_m, columns * rows * sizeof(double));
    cudaMemcpy(d_m_gpu, src.d_m_gpu, columns * rows * sizeof(double), cudaMemcpyDeviceToDevice);
}

// --- Operator Overloads executing parallel operations via Kernels ---

Matrix Matrix::operator+(const Matrix& other) const {
    assert(rows == other.rows && columns == other.columns);
    Matrix res(rows, columns);
    int size = rows * columns;
    int gridDim = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    matrix_sum_kernel<<<gridDim, THREADS_PER_BLOCK>>>(d_m_gpu, other.d_m_gpu, res.d_m_gpu, size);
    return res; 
}

Matrix Matrix::operator-(const Matrix& other) const {
    assert(rows == other.rows && columns == other.columns);
    Matrix res(rows, columns);
    int size = rows * columns;
    int gridDim = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    matrix_minus_kernel<<<gridDim, THREADS_PER_BLOCK>>>(d_m_gpu, other.d_m_gpu, res.d_m_gpu, size);
    return res;
}

Matrix Matrix::operator*(const Matrix& other) const {
    assert(columns == other.rows);
    Matrix res(rows, other.columns);

    dim3 blockDim(16, 16);
    dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (other.columns + blockDim.y - 1) / blockDim.y);

    matrix_dot_kernel<<<gridDim, blockDim>>>(d_m_gpu, other.d_m_gpu, res.d_m_gpu, rows, columns, other.columns);
    return res;
}

Matrix Matrix::hadamard(const Matrix& other) const {
    assert(rows == other.rows && columns == other.columns);
    Matrix res(rows, columns);
    int size = rows * columns;
    int gridDim = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    matrix_hadamard_kernel<<<gridDim, THREADS_PER_BLOCK>>>(d_m_gpu, other.d_m_gpu, res.d_m_gpu, size);
    return res;
}

Matrix Matrix::apply_function(func_id_t f) const {
    Matrix res(rows, columns);
    int size = rows * columns;
    int gridDim = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    matrix_function_kernel<<<gridDim, THREADS_PER_BLOCK>>>(d_m_gpu, f, res.d_m_gpu, size);
    return res;
}

Matrix Matrix::transpose() const {
    Matrix res(columns, rows);
    dim3 blockDim(16, 16);
    dim3 gridDim((rows + blockDim.x - 1) / blockDim.x, (columns + blockDim.y - 1) / blockDim.y);

    matrix_transpose_kernel<<<gridDim, blockDim>>>(d_m_gpu, res.d_m_gpu, rows, columns);
    return res;
}

Matrix Matrix::scale(double s) const {
    Matrix res(rows, columns);
    int size = rows * columns;
    int gridDim = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    matrix_scalar_kernel<<<gridDim, THREADS_PER_BLOCK>>>(d_m_gpu, s, res.d_m_gpu, size);
    return res;
}

void Matrix::print(bool is_short) const {
    unsigned lim_rows = is_short ? std::min(rows, 4u) : rows;
    unsigned lim_col = is_short ? std::min(columns, 10u) : columns;

    for (unsigned row = 0; row < lim_rows; row++) {
        for (unsigned col = 0; col < lim_col; col++) {
            printf("%.2lf ", h_m[col + row * columns]);
        }
        if (is_short && lim_col != columns) printf("...");
        printf("\n");
    }
    if (is_short && lim_rows != rows) printf("...\n");
}