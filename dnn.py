import numpy as np
from scipy import special


class DNN:

    def __init__(self, layers: list, learning_rate: float):
        """
        initialise the neural network
        :param layers: node numbers in each layer
        :param learning_rate: learning rate
        """
        self.layers = layers
        self.lr = learning_rate
        # network weights, biases
        self.weights = []
        self.biases = []
        for i in range(len(self.layers) - 1):
            self.weights.append(np.random.rand(self.layers[i + 1], self.layers[i]) - 0.5)
            self.biases.append(np.random.rand(self.layers[i + 1], 1) - 0.5)
        self.activation_f = lambda x: special.expit(x)

    def train(self, inputs_list, target_list):
        """
        train the neural network
        :param inputs_list: inputs of neural network
        :param target_list:
        :return: loss
        """
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        outputs = []  # outputs of each layer
        x = inputs
        for w, b in zip(self.weights, self.biases):
            layer_inputs = np.dot(w, x) + b
            layer_outputs = self.activation_f(layer_inputs)
            outputs.append(x)
            x = layer_outputs
        outputs.append(x)

        errors = targets - outputs[-1]  # final errors
        loss = np.power(errors, 2).sum()
        for i in range(len(outputs) - 1, 0, -1):
            # update the weights
            # delta_W_jk = alpha * Ek * Ok * (1 - Ok) @ Oj.T
            # delta_B_k = alpha * Ek * Ok * (1 - Ok)
            b_delta = errors * outputs[i] * (1 - outputs[i])
            self.biases[i - 1] += self.lr * b_delta
            self.weights[i - 1] += self.lr * np.dot(b_delta, outputs[i - 1].T)
            errors = np.dot(self.weights[i - 1].T, errors)  # back propagating errors

        return loss

    def forward(self, inputs_list):
        """
        query the neural network
        :param inputs_list: inputs of neural network
        :return: outputs of neural network
        """
        x = np.array(inputs_list, ndmin=2).T  # convert inputs to 2D
        for w, b in zip(self.weights, self.biases):
            x = np.dot(w, x) + b
            x = self.activation_f(x)
        return x
