import numpy as np
from scipy import special


class DNN:

    def __init__(self, layers, learning_rate):
        self.layers = layers
        self.lr = learning_rate

        # self.input_hidden_w = np.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
        # self.hidden_output_w = np.random.rand(self.output_nodes, self.hidden_nodes) - 0.5
        self.weights = []
        for i in range(len(self.layers) - 1):
            self.weights.append(np.random.rand(self.layers[i + 1], self.layers[i]) - 0.5)
        self.activation_f = lambda x: special.expit(x)

    def train(self, inputs_list, target_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        # hidden_inputs = np.dot(self.input_hidden_w, inputs)
        # hidden_outputs = self.activation_f(hidden_inputs)
        # final_inputs = np.dot(self.hidden_output_w, hidden_outputs)
        # final_outputs = self.activation_f(final_inputs)
        outputs = []
        x = inputs
        for w in self.weights:
            layer_inputs = np.dot(w, x)
            layer_outputs = self.activation_f(layer_inputs)
            outputs.append(x)
            # outputs.append(layer_outputs)
            x = layer_outputs
        outputs.append(x)
        # output_errors = targets - final_outputs
        # hidden_errors = np.dot(self.hidden_output_w.T, output_errors)
        # self.hidden_output_w += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)),
        #                                          hidden_outputs.T)
        # self.input_hidden_w += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), inputs.T)

        errors = targets - outputs[-1]
        for i in range(len(outputs) - 1, 0, -1):
            self.weights[i - 1] += self.lr * np.dot((errors * outputs[i] * (1 - outputs[i])), outputs[i - 1].T)
            errors = np.dot(self.weights[i - 1].T, errors)

    def forward(self, inputs_list):
        x = np.array(inputs_list, ndmin=2).T
        # hidden_inputs = np.dot(self.input_hidden_w, inputs)
        # hidden_outputs = self.activation_f(hidden_inputs)
        # final_inputs = np.dot(self.hidden_output_w, hidden_outputs)
        # final_outputs = self.activation_f(final_inputs)
        # return final_outputs
        for w in self.weights:
            x = np.dot(w, x)
            x = self.activation_f(x)
        return x
