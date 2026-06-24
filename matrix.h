#ifndef MATRIX_H
#define MATRIX_H

#include <cuda_runtime.h>
#include <iostream>

// Enum matching your original design for activation functions
enum func_id_t {
    FUNC_ADDITION,
    FUNC_SIGMOID,
    FUNC_RELU,
    FUNC_DSIGMOID
};

class Matrix {
private:
    unsigned rows;
    unsigned columns;
    double* h_m;     // Host Pointer
    double* d_m_gpu; // Device Pointer

public:
    // Lifecycle Management (RAII)
    Matrix(unsigned r, unsigned c);
    ~Matrix();

    // Prevent dangerous shallow copies (Double Free Vulnerability)
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;

    // Enable high-performance move semantics
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(Matrix&& other) noexcept;

    // Synchronizations
    void CPU_to_GPU();
    void GPU_to_CPU();
    void print(bool is_short = true) const;

    // High-Performance Utility Operators
    double& operator()(unsigned r, unsigned c) { return h_m[r * columns + c]; }
    const double& operator()(unsigned r, unsigned c) const { return h_m[r * columns + c]; }
    void copy_from(const Matrix& src); // Replaces matrix_memcpy

    // Encapsulated Mathematical GPU Interfaces
    Matrix operator+(const Matrix& other) const;            // matrix_sum_gpu
    Matrix operator-(const Matrix& other) const;            // matrix_minus_gpu
    Matrix operator*(const Matrix& other) const;            // matrix_dot_gpu (Matrix Multiplication)
    Matrix hadamard(const Matrix& other) const;             // matrix_hadamard_gpu
    Matrix apply_function(func_id_t f) const;               // matrix_function_gpu
    Matrix transpose() const;                               // matrix_transpose_gpu
    Matrix scale(double s) const;                           // matrix_scalar_gpu

    // In-place GPU operations: write the result into this matrix's existing
    // buffer (no allocation). Used on the hot path (forward/backward) to keep
    // the training loop free of cudaMalloc/cudaFree.
    void set_sum(const Matrix& a, const Matrix& b);       // this <- a + b
    void set_sub(const Matrix& a, const Matrix& b);       // this <- a - b
    void set_hadamard(const Matrix& a, const Matrix& b);  // this <- a o b
    void set_scale(const Matrix& a, double s);            // this <- a * s
    void set_apply(const Matrix& a, func_id_t f);         // this <- f(a)
    void set_dot(const Matrix& a, const Matrix& b);       // this <- a . b (matmul)
    void set_transpose(const Matrix& a);                  // this <- a^T

    // Core Getters for custom Kernel/Layer interaction
    double* gpu_ptr() const { return d_m_gpu; }
    double* cpu_ptr() const { return h_m; }
    unsigned get_rows() const { return rows; }
    unsigned get_cols() const { return columns; }
};

#endif