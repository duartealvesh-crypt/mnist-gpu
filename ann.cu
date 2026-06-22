#include "ann.h"
#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>
#include <stdint.h>

double normalRand(double mu, double sigma);
void init_weight(matrix_t* w, unsigned nneurones_prev);
void print_layer(layer_t *layer);


// Standard Box-Muller transform: draws one sample from N(mu, sigma^2).
double normalRand(double mu, double sigma)
{
	const double epsilon = DBL_MIN;
	const double two_pi = 2.0*M_PI;

	double u1, u2;
	do
	 {
	   u1 = (double) rand() / RAND_MAX;
	   u2 = (double) rand() / RAND_MAX;
	 }
	while ( u1 <= epsilon );

	double z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	return z0 * sigma + mu;
}

void init_weight(matrix_t* w, unsigned nneurones_prev)
{
    for (int idx = 0; idx < w->columns * w->rows; idx ++)
    {
        w->m[idx] = normalRand(0, 1 / sqrt(nneurones_prev));
    }
    // Weights are initialized once on the CPU, then pushed to the GPU a single time before training starts.
    matrix_CPU_to_GPU(w);
}

ann_t * create_ann(double alpha, unsigned minibatch_size, unsigned number_of_layers, unsigned* nneurons_per_layer)
{
    ann_t * nn = (ann_t *)malloc(sizeof(ann_t));

    nn->layers = (layer_t **)malloc(number_of_layers * sizeof(layer_t *));
    nn->number_of_layers = number_of_layers;
    nn->alpha = alpha;
    nn->minibatch_size = minibatch_size;

    nn->layers[0] = create_layer(0, nneurons_per_layer[0], minibatch_size, minibatch_size);
    for (int l = 1; l < number_of_layers; l++)
    {
        nn->layers[l] = create_layer(l, nneurons_per_layer[l], nneurons_per_layer[l-1], minibatch_size);
    }

    return nn;
}

layer_t * create_layer(unsigned layer_number, unsigned number_of_neurons, unsigned nneurons_previous_layer, unsigned minibatch_size)
{
    layer_t * layer = (layer_t*) malloc(sizeof(layer_t));

    layer->number_of_neurons = number_of_neurons;
    layer->minibatch_size = minibatch_size;    
    layer->activations = alloc_matrix(number_of_neurons, minibatch_size);
    layer->z = alloc_matrix(number_of_neurons, minibatch_size);
    layer->delta = alloc_matrix(number_of_neurons, minibatch_size);
    layer->weights = alloc_matrix(number_of_neurons, nneurons_previous_layer);    
    layer->biases = alloc_matrix(number_of_neurons, 1);

    if (layer_number > 0)
    {
        init_weight(layer->weights, nneurons_previous_layer);
    }

    layer->one_col = alloc_matrix(minibatch_size, 1);
    for (int i = 0; i < (int) minibatch_size; i++)
        layer->one_col->m[i] = 1.0;
    matrix_CPU_to_GPU(layer->one_col);

    layer->one_row = alloc_matrix(1, minibatch_size);
    for (int i = 0; i < (int) minibatch_size; i++)
        layer->one_row->m[i] = 1.0;
    matrix_CPU_to_GPU(layer->one_row);
    
    // Forward scratch matrices
    layer->z1_tmp = alloc_matrix(number_of_neurons, minibatch_size); // W x a^(l-1)
    layer->z2_tmp = alloc_matrix(number_of_neurons, minibatch_size); // b broadcast
    // Backward scratch matrices
    layer->tw        = alloc_matrix(nneurons_previous_layer, number_of_neurons);
    layer->delta_tmp = alloc_matrix(nneurons_previous_layer, minibatch_size);
    layer->dfz       = alloc_matrix(nneurons_previous_layer, minibatch_size);
    layer->ta        = alloc_matrix(minibatch_size, nneurons_previous_layer);
    layer->w1        = alloc_matrix(number_of_neurons, nneurons_previous_layer);
    layer->b1        = alloc_matrix(number_of_neurons, 1);



    return layer;
}

void set_input(ann_t *nn, matrix_t* input){
    // Copy the minibatch into the input layer once per epoch, then push it to the
    // GPU so the rest of the forward pass can run entirely on device memory.
    matrix_memcpy(nn->layers[0]->activations, input);
    matrix_CPU_to_GPU(nn->layers[0]->activations);
}

void print_layer(layer_t *layer)
{
    printf("-- neurons:%d, minibatch size:%d\n", layer->number_of_neurons, layer->minibatch_size);

    printf(">> Weighted inputs --\n");
    print_matrix(layer->z, true);
    printf(">> Activations --\n");
    print_matrix(layer->activations, true);
    
    printf(">> Weights --\n");
    print_matrix(layer->weights, true);
    printf(">> Biases --\n");
    print_matrix(layer->biases, true);

    printf(">> Delta --\n");
    print_matrix(layer->delta, true);
    
}

void print_nn(ann_t *nn)
{
    printf("ANN -- nlayers:%d, alpha:%lf, minibatch size: %d\n", nn->number_of_layers, nn->alpha, nn->minibatch_size);
    for (int l = 0; l < nn->number_of_layers; l++)
    {
        printf("Layer %d ", l);
        print_layer(nn->layers[l]);
    }
}

void forward(ann_t *nn, double (*activation_function)(double))
{
    for (int l = 1; l < nn->number_of_layers; l++)
    {
        layer_t *cur  = nn->layers[l];
        layer_t *prev = nn->layers[l - 1];


        matrix_dot_gpu(cur->weights, prev->activations, cur->z1_tmp); // z1 = W^l x a^(l-1)
        matrix_dot_gpu(cur->biases, cur->one_row, cur->z2_tmp); // z2 = b^l x [1,1,...,1]
        matrix_sum_gpu(cur->z1_tmp, cur->z2_tmp, cur->z); // z^l = z1 + z2
        matrix_function_gpu(cur->z, FUNC_SIGMOID, cur->activations); // a^l = sigmoid(z^l)
    }
}

void backward(ann_t *nn, matrix_t *y, double (*derivative_actfunct)(double))
{
    unsigned L = nn->number_of_layers-1;
    // Output layer delta is computed separately since it doesn't follow the
    // recurrence used for hidden layers below.
    matrix_t *dfzL = alloc_matrix(nn->layers[L]->number_of_neurons, nn->minibatch_size);

    matrix_minus_gpu(nn->layers[L]->activations, y, nn->layers[L]->delta);  // delta^(L) = (a^L - y)
    matrix_function_gpu(nn->layers[L]->z, FUNC_DSIGMOID, dfzL); // f'(z^(L))
    matrix_hadamard_gpu(nn->layers[L]->delta, dfzL, nn->layers[L]->delta); // delta^(L) = (a^L - y) o f'(z^(L))

    destroy_matrix(dfzL);

    for (int l = L; l > 1; l--)
    {
        layer_t *cur  = nn->layers[l];
        layer_t *prev = nn->layers[l - 1];
        
        matrix_transpose_gpu(cur->weights, cur->tw); // (w^l)T        
        matrix_dot_gpu(cur->tw, cur->delta, cur->delta_tmp); // (w^l)T x delta^l
        matrix_function_gpu(prev->z, FUNC_DSIGMOID, cur->dfz); // f'(z^(l-1))
        matrix_hadamard_gpu(cur->delta_tmp, cur->dfz, prev->delta); // delta^(l-1) = (w^l)T x delta^l o f'(z^(l-1))
    }
    
    for (int l = 1; l < nn->number_of_layers; l++)
    {
        layer_t *cur  = nn->layers[l];
        layer_t *prev = nn->layers[l - 1];

        matrix_transpose_gpu(prev->activations, cur->ta); // ta <- (a^(l-1))^T
        matrix_dot_gpu(cur->delta, cur->ta, cur->w1); // w1 <- delta^l x (a^(l-1))^T
        matrix_scalar_gpu(cur->w1, nn->alpha / nn->minibatch_size, cur->w1);// w1 <- alpha /m . delta^l x (a^(l-1))^T
        matrix_minus_gpu(cur->weights, cur->w1, cur->weights); // w^l <- w^l - alpha /m . delta^l x (a^(l-1))^T



        matrix_dot_gpu(cur->delta, cur->one_col, cur->b1); // b1 <- delta^l x 1^T
        matrix_scalar_gpu(cur->b1, nn->alpha / nn->minibatch_size, cur->b1); // b1 <- alpha / m . delta^l x 1^T
        matrix_minus_gpu(cur->biases, cur->b1, cur->biases); // b^l = b^l - alpha / m . delta^l x 1^T

    }
}

void destroy_layer(layer_t *layer)
{
    // Matrices permanentes
    destroy_matrix(layer->weights);
    destroy_matrix(layer->biases);
    destroy_matrix(layer->z);
    destroy_matrix(layer->activations);
    destroy_matrix(layer->delta);

    // Forward scratch matrices
    destroy_matrix(layer->z1_tmp);
    destroy_matrix(layer->z2_tmp);

    // Backward scratch matrices
    destroy_matrix(layer->tw);
    destroy_matrix(layer->delta_tmp);
    destroy_matrix(layer->dfz);
    destroy_matrix(layer->ta);
    destroy_matrix(layer->w1);
    destroy_matrix(layer->b1);
    destroy_matrix(layer->one_row);
    destroy_matrix(layer->one_col);


    free(layer);
}

void destroy_ann(ann_t *nn)
{
    for (unsigned l = 0; l < nn->number_of_layers; l++)
        destroy_layer(nn->layers[l]);
    free(nn->layers);
    free(nn);
}

