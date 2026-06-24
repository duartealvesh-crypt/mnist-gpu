#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "mnist.h"
#include "matrix.h"
#include "ann_cpp.h"

void populate_minibatch(double *x, double* y, unsigned* minibatch_idx, unsigned minibatch_size, image * img, unsigned img_size, byte* label, unsigned label_size);

void zero_to_n(unsigned n, unsigned* t) {
    for (unsigned i = 0; i < n; i++) t[i] = i;
}

void shuffle(unsigned *t, const unsigned size, const unsigned number_of_switch) {
    zero_to_n(size, t);
    for (unsigned i = 0; i < number_of_switch; i++) {
        unsigned x = rand() % size;
        unsigned y = rand() % size;
        unsigned tmp = t[x];
        t[x] = t[y];
        t[y] = tmp;
    }
}

double accuracy(image* test_img, byte* test_label, unsigned datasize, unsigned minibatch_size, Ann& nn) {
    unsigned good = 0;
    unsigned *idx = (unsigned *) malloc(datasize * sizeof(unsigned));
    
    // Objets Matrix temporaires pour manipuler les données proprement
    Matrix in_batch(28 * 28, minibatch_size);
    
    zero_to_n(datasize, idx);
    unsigned L = nn.get_number_of_layers() - 1; // Utilisation d'un getter pour la sécurité

    for (unsigned i = 0; i < datasize - minibatch_size; i += minibatch_size) {
        // Remplissage direct du tableau CPU de notre matrice d'entrée
        populate_minibatch(in_batch.cpu_ptr(), nullptr, &idx[i], minibatch_size, test_img, 28*28, test_label, 10);

        in_batch.CPU_to_GPU();
        nn.set_input(in_batch);

        nn.forward(FUNC_SIGMOID);

        // On rapatrie la couche de sortie sur le CPU pour faire la vérification
        nn.get_layer(L).get_activations().GPU_to_CPU(); 
        double* act_cpu = nn.get_layer(L).get_activations().cpu_ptr();

        for (unsigned col = 0; col < minibatch_size; col++) {
            unsigned idxTrainingData = col + i;
            double max_val = -1.0;
            unsigned idx_max = 0;
            
            for (unsigned row = 0; row < 10; row++) {
                unsigned idx_mat = col + row * minibatch_size;
                if (act_cpu[idx_mat] > max_val) {
                    max_val = act_cpu[idx_mat];
                    idx_max = row;
                }
            }
            if (idx_max == test_label[idxTrainingData]) {
                good++;
            }
        }
    }
    free(idx);

    unsigned ntests = (datasize / minibatch_size) * minibatch_size;
    return (100.0 * (double)(good) / ntests);
}

void populate_minibatch(double * x, double * y, unsigned * minibatch_idx, unsigned minibatch_size, image * img, unsigned img_size, byte* label, unsigned label_size) {
    for (unsigned col = 0; col < minibatch_size; col++) {
        if (x) {
            for (unsigned row = 0; row < img_size; row++) {
                x[row * minibatch_size + col] = (double) img[minibatch_idx[col]][row] / 255.0;
            }
        }
        if (y) {
            for (unsigned row = 0; row < 10; row++) {
                y[row * minibatch_size + col] = 0.0;
            }
            y[label[minibatch_idx[col]] * minibatch_size + col] = 1.0;
        }
    }
}

int main(int argc, char *argv[]) {
    srand(time(0));
    unsigned datasize, ntest;
    image* train_img = read_images("train-images.idx3-ubyte", &datasize);
    byte* train_label = read_labels("train-labels.idx1-ubyte", &datasize);
    image* test_img = read_images("t10k-images.idx3-ubyte", &ntest);
    byte* test_label = read_labels("t10k-labels.idx1-ubyte", &ntest);

    double alpha = 0.05;
    unsigned minibatch_size = 16;
    unsigned number_of_layers = 3;
    unsigned nneurons_per_layer[3] = {28*28, 30, 10};
    
    Ann nn(alpha, minibatch_size, number_of_layers, nneurons_per_layer);

    printf("Starting accuracy: %lf%%\n", accuracy(test_img, test_label, ntest, minibatch_size, nn));

    unsigned *shuffled_idx = (unsigned *)malloc(datasize * sizeof(unsigned));
    Matrix in(28 * 28, minibatch_size);
    Matrix out(10, minibatch_size);

    for (int epoch = 0; epoch < 5; epoch++) {
        printf("Start learning epoch %d\n", epoch);
        shuffle(shuffled_idx, datasize, datasize);

        for (unsigned i = 0; i < datasize - minibatch_size; i += minibatch_size) {
            // Remplissage des tableaux CPU internes des matrices
            populate_minibatch(in.cpu_ptr(), out.cpu_ptr(), shuffled_idx + i, minibatch_size, train_img, 28*28, train_label, 10);

            // Envoi des données sur le GPU
            in.CPU_to_GPU();
            out.CPU_to_GPU();
            
            // On injecte l'entrée dans le réseau
            nn.set_input(in);

            // Cycle Forward + Backward complet sur GPU
            nn.forward(FUNC_SIGMOID);
            nn.backward(out, FUNC_DSIGMOID);
        }
        
        printf("Epoch %d accuracy: %lf%%\n", epoch, accuracy(test_img, test_label, ntest, minibatch_size, nn));
    }

    free(shuffled_idx);
    return 0;
}