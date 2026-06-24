#ifndef ANN_H
#define ANN_H

#include <cuda_runtime.h>
#include <iostream>
#include "matrix.h"
#include <vector>

class Layer {
private:
    unsigned minibatch_size;
    unsigned number_of_neurons;
    
    Matrix weights;
    Matrix biases;

    Matrix z;
    Matrix activations;
    
    Matrix delta;
    // forward scratch matrices
    Matrix z1_tmp;
    Matrix z2_tmp;
    // backward scratch matrices
    Matrix tw;
    Matrix delta_tmp;
    Matrix dfz;
    Matrix ta;
    Matrix w1;
    Matrix b1;
    Matrix one_col;
    Matrix one_row;


public:
    // Lifecycle Management (RAII)
    Layer(unsigned batch_size, unsigned neurons);
    ~Layer();

    // Prevent dangerous shallow copies (Double Free Vulnerability)
    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;

    // Enable high-performance move semantics
    Layer(Layer&& other) noexcept;
    Layer& operator=(Layer&& other) noexcept;

    void print_layer();

};


class Ann {
private:
    void (*f)(double*, double*, unsigned, unsigned);
    void (*fd)(double*, double*, unsigned, unsigned);
    double alpha;
    unsigned minibatch_size;
    unsigned input_size;
    unsigned number_of_layers;
    std::vector<Layer> layers;

public:
    // Lifecycle Management (RAII)
    Ann(double alpha, unsigned minibatch_size, unsigned number_of_layers, unsigned* nneurons_per_layer);
    ~Ann();

    // Prevent dangerous shallow copies (Double Free Vulnerability)
    Ann(const Ann&) = delete;
    Ann& operator=(const Ann&) = delete;

    // Enable high-performance move semantics
    Ann(Ann&& other) noexcept;
    Ann& operator=(Ann&& other) noexcept;

    void set_input(Matrix* input);
    void print_nn();
    void forward(func_id_t activation_type);
    void bacward(Matrix *y, func_id_t activation_derivative_type);

};

#endif