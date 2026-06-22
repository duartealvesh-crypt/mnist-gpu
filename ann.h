#ifndef ANN_H
#define ANN_H
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "matrix.h"

typedef struct 
{
    unsigned minibatch_size;
    unsigned number_of_neurons;
    
    matrix_t* weights;
    matrix_t* biases;

    matrix_t* z;
    matrix_t* activations;
    
    matrix_t* delta;
    // forward scratch matrices
    matrix_t* z1_tmp;
    matrix_t* z2_tmp;
    // backward scratch matrices
    matrix_t* tw;
    matrix_t* delta_tmp;
    matrix_t* dfz;
    matrix_t* ta;
    matrix_t* w1;
    matrix_t* b1;
    matrix_t* one_col;
    matrix_t* one_row;

} layer_t;

typedef struct 
{
    void (*f)(double*, double*, unsigned, unsigned);
    void (*fd)(double*, double*, unsigned, unsigned);
    double alpha;
    unsigned minibatch_size;
    unsigned input_size;
    unsigned number_of_layers;
    layer_t** layers;
} ann_t;

ann_t * create_ann(double alpha, unsigned minibatch_size, unsigned number_of_layers, unsigned* nneurons_per_layer);

layer_t * create_layer(unsigned l, unsigned number_of_neurons, unsigned nneurons_previous_layer, unsigned minibatch_size);

void set_input(ann_t *nn, matrix_t* input);

void print_nn(ann_t *nn);

void forward(ann_t *nn, double (*activation_function)(double));

void backward(ann_t *nn, matrix_t *y, double (*derivative_actfunct)(double));

#endif