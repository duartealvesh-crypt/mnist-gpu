#ifndef ANN_CPP_H
#define ANN_CPP_H

#include <vector>
#include "matrix.h"

// A single fully-connected layer. All matrices (weights, activations and the
// forward/backward scratch buffers) live as members so they are allocated once,
// at construction, and reused across the whole training run.
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

    // Ann drives the forward/backward passes directly over a Layer's matrices.
    friend class Ann;

public:
    Layer(unsigned layer_number, unsigned number_of_neurons,
          unsigned nneurons_previous_layer, unsigned minibatch_size);
    ~Layer() = default;

    // No copies (each Matrix owns a GPU buffer); moves only.
    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;
    Layer(Layer&&) noexcept = default;
    Layer& operator=(Layer&&) noexcept = default;

    Matrix& get_activations() { return activations; }
    void print_layer();
};

// The network: owns its layers and runs forward/backward entirely on the GPU.
class Ann {
private:
    double alpha;
    unsigned minibatch_size;
    unsigned number_of_layers;
    std::vector<Layer> layers;

public:
    Ann(double alpha, unsigned minibatch_size, unsigned number_of_layers,
        unsigned* nneurons_per_layer);
    ~Ann() = default;

    Ann(const Ann&) = delete;
    Ann& operator=(const Ann&) = delete;
    Ann(Ann&&) noexcept = default;
    Ann& operator=(Ann&&) noexcept = default;

    unsigned get_number_of_layers() const { return number_of_layers; }
    Layer& get_layer(unsigned i) { return layers[i]; }

    void set_input(Matrix& input);
    void forward(func_id_t activation_type);
    void backward(Matrix& y, func_id_t activation_derivative_type);
    void print_nn();
};

#endif
