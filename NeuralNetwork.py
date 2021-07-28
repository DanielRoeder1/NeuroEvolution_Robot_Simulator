import numpy as np
import random

class NeuralNetwork2:
    def __init__(self, layer_size, bit_representation , input_size, delta_cycle_nn):
        self.num_layers = len(layer_size)
        self.layer_size = layer_size
        self.recurrent_input = [0,0,0,0]
        self.bit_representation = bit_representation
        self.input_size = input_size
        self.delta_cycle_nn = delta_cycle_nn

    def calculate(self, initial_input, gene, cycle_num):
        # deduct half of the maximum value stored by bits to go negative
        half_max = sum(2 ** (np.arange(self.bit_representation))) // 2
        layer_input = self.decay_distance(np.array(initial_input).clip(0))
        layer_input.extend(self.recurrent_input)
        # Needed to select layer weights from gene
        gene_pointer = [0, 0]
        for layer_size in self.layer_size:
            input_size = len(layer_input)
            gene_pointer[1] += input_size * layer_size * self.bit_representation + layer_size * self.bit_representation

            layer_bits = np.array(gene[gene_pointer[0]:gene_pointer[1]])
            bit_weights = layer_bits[:-(layer_size * self.bit_representation)].reshape(layer_size, input_size,self.bit_representation)
            bit_biases = layer_bits[-(layer_size * self.bit_representation):].reshape(layer_size, self.bit_representation)
            layer_biases = bit_biases.dot(2 ** np.arange(self.bit_representation)[::-1])
            layer_weights = bit_weights.dot(2 ** np.arange(self.bit_representation)[::-1]).T

            layer_weights = (layer_weights - half_max) / 10
            layer_biases = (layer_biases - half_max) / 10


            gene_pointer[0] = gene_pointer[1]
            layer_input = np.dot(layer_input, layer_weights) + layer_biases
            layer_input = self.sigmoid_activation(layer_input)

            ##### Experiment with different cycle numbers #####
            if cycle_num % self.delta_cycle_nn == 0 and layer_size == 4:
                self.recurrent_input = layer_input


        return layer_input

    def sigmoid_activation(self, input):
        return 1 / (1 + np.exp(-input))

    def create_genomes(self, num_individuals):
        num_bits = 0
        # Include recurrent part
        input_size = self.input_size + self.layer_size[0]
        for layer in self.layer_size:
                        # Weights                                    #Biases
            num_bits += input_size * layer * self.bit_representation + layer * self.bit_representation
            input_size = layer

        genomes = np.zeros((num_individuals, num_bits), dtype=int)
        # Random initialization
        for genome in genomes:
            for i in range(num_bits):
                genome[i] = random.randint(0, 1)

        return genomes

    def decay_distance(self,initial_intput):
        A = 20
        alpha = 0.1
        t = 5
        return list(A + (A * alpha - A) * (1 - np.exp(-np.array(initial_intput) / t)))

