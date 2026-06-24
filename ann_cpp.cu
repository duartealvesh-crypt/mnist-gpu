#include "ann_cpp.h"
#include "matrix.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>

double normalRand(double mu, double sigma);
void init_weight(Matrix& w, unsigned nneurons_prev);

// Standard Box-Muller transform: draws one sample from N(mu, sigma^2).
double normalRand(double mu, double sigma)
{
    const double epsilon = DBL_MIN;
    const double two_pi = 2.0 * M_PI;

    double u1, u2;
    do {
        u1 = (double) rand() / RAND_MAX;
        u2 = (double) rand() / RAND_MAX;
    } while (u1 <= epsilon);

    double z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
    return z0 * sigma + mu;
}

void init_weight(Matrix& w, unsigned nneurons_prev)
{
    for (unsigned r = 0; r < w.get_rows(); ++r)
        for (unsigned c = 0; c < w.get_cols(); ++c)
            w(r, c) = normalRand(0, 1.0 / sqrt(nneurons_prev));
    // Initialized once on the CPU, then pushed to the GPU before training starts.
    w.CPU_to_GPU();
}

Layer::Layer(unsigned layer_number, unsigned number_of_neurons,
             unsigned nneurons_previous_layer, unsigned minibatch_size)
    : minibatch_size(minibatch_size),
      number_of_neurons(number_of_neurons),
      weights(number_of_neurons, nneurons_previous_layer),
      biases(number_of_neurons, 1),
      z(number_of_neurons, minibatch_size),
      activations(number_of_neurons, minibatch_size),
      delta(number_of_neurons, minibatch_size),
      z1_tmp(number_of_neurons, minibatch_size),
      z2_tmp(number_of_neurons, minibatch_size),
      tw(nneurons_previous_layer, number_of_neurons),
      delta_tmp(nneurons_previous_layer, minibatch_size),
      dfz(nneurons_previous_layer, minibatch_size),
      ta(minibatch_size, nneurons_previous_layer),
      w1(number_of_neurons, nneurons_previous_layer),
      b1(number_of_neurons, 1),
      one_col(minibatch_size, 1),
      one_row(1, minibatch_size)
{
    if (layer_number > 0) {
        init_weight(weights, nneurons_previous_layer);
    }

    for (unsigned i = 0; i < minibatch_size; ++i) {
        one_col(i, 0) = 1.0;
        one_row(0, i) = 1.0;
    }
    one_col.CPU_to_GPU();
    one_row.CPU_to_GPU();
}

Ann::Ann(double alpha, unsigned minibatch_size, unsigned number_of_layers,
         unsigned* nneurons_per_layer)
    : alpha(alpha),
      minibatch_size(minibatch_size),
      number_of_layers(number_of_layers)
{
    layers.reserve(number_of_layers);
    // Layer 0 is the input layer: it has no real weights, so its "previous layer"
    // size is irrelevant (set to minibatch_size, matching the original design).
    layers.emplace_back(0, nneurons_per_layer[0], minibatch_size, minibatch_size);
    for (unsigned l = 1; l < number_of_layers; ++l) {
        layers.emplace_back(l, nneurons_per_layer[l], nneurons_per_layer[l - 1], minibatch_size);
    }
}

void Ann::set_input(Matrix& input)
{
    // Copy the staged minibatch into the input layer's activations (host + device).
    layers[0].activations.copy_from(input);
}

void Ann::forward(func_id_t activation_type)
{
    for (unsigned l = 1; l < number_of_layers; ++l) {
        Layer& cur  = layers[l];
        Layer& prev = layers[l - 1];

        cur.z1_tmp      = cur.weights * prev.activations;        // z1 = W^l x a^(l-1)
        cur.z2_tmp      = cur.biases * cur.one_row;              // z2 = b^l x [1,...,1]
        cur.z           = cur.z1_tmp + cur.z2_tmp;              // z^l = z1 + z2
        cur.activations = cur.z.apply_function(activation_type); // a^l = sigmoid(z^l)
    }
}

void Ann::backward(Matrix& y, func_id_t d_activation_type)
{
    unsigned L = number_of_layers - 1;

    // Output layer delta: (a^L - y) o f'(z^L)
    layers[L].delta = layers[L].activations - y;
    layers[L].dfz   = layers[L].z.apply_function(d_activation_type);
    layers[L].delta = layers[L].delta.hadamard(layers[L].dfz);

    // Propagate the delta back through the hidden layers.
    for (unsigned l = L; l > 1; --l) {
        Layer& cur  = layers[l];
        Layer& prev = layers[l - 1];

        cur.tw        = cur.weights.transpose();                  // (W^l)^T
        cur.delta_tmp = cur.tw * cur.delta;                       // (W^l)^T x delta^l
        prev.dfz      = prev.z.apply_function(d_activation_type); // f'(z^(l-1))
        prev.delta    = cur.delta_tmp.hadamard(prev.dfz);         // delta^(l-1)
    }

    // Apply the gradients: W <- W - (alpha/m) * delta x a^T,  b <- b - (alpha/m) * delta x 1
    double lr = alpha / minibatch_size;
    for (unsigned l = 1; l < number_of_layers; ++l) {
        Layer& cur  = layers[l];
        Layer& prev = layers[l - 1];

        cur.ta      = prev.activations.transpose();        // (a^(l-1))^T
        cur.w1      = cur.delta * cur.ta;                  // delta^l x (a^(l-1))^T
        cur.weights = cur.weights - cur.w1.scale(lr);      // weight update

        cur.b1      = cur.delta * cur.one_col;             // delta^l summed over the batch
        cur.biases  = cur.biases - cur.b1.scale(lr);       // bias update
    }
}

void Layer::print_layer()
{
    printf("-- neurons:%u, minibatch size:%u\n", number_of_neurons, minibatch_size);
    printf(">> Weighted inputs --\n"); z.print(true);
    printf(">> Activations --\n");     activations.print(true);
    printf(">> Weights --\n");         weights.print(true);
    printf(">> Biases --\n");          biases.print(true);
    printf(">> Delta --\n");           delta.print(true);
}

void Ann::print_nn()
{
    printf("ANN -- nlayers:%u, alpha:%lf, minibatch size:%u\n",
           number_of_layers, alpha, minibatch_size);
    for (unsigned l = 0; l < number_of_layers; ++l) {
        printf("Layer %u ", l);
        layers[l].print_layer();
    }
}
