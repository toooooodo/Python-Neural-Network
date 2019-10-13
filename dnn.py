import numpy as np
from scipy import special


class DNN:

    def __init__(self, input_n, hidden_n, output_n, learning_rate):
        self.input_nodes = input_n
        self.hidden_nodes = hidden_n
        self.output_nodes = output_n
        self.lr = learning_rate

        self.input_hidden_w = np.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
        self.hidden_output_w = np.random.rand(self.output_nodes, self.hidden_nodes) - 0.5

        self.activation_f = lambda x: special.expit(x)

    def train(self, inputs_list, target_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        hidden_inputs = np.dot(self.input_hidden_w, inputs)
        hidden_outputs = self.activation_f(hidden_inputs)
        final_inputs = np.dot(self.hidden_output_w, hidden_outputs)
        final_outputs = self.activation_f(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.hidden_output_w.T, output_errors)
        self.hidden_output_w += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)),
                                                 hidden_outputs.T)
        self.input_hidden_w += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), inputs.T)

    def forward(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.input_hidden_w, inputs)
        hidden_outputs = self.activation_f(hidden_inputs)
        final_inputs = np.dot(self.hidden_output_w, hidden_outputs)
        final_outputs = self.activation_f(final_inputs)
        return final_outputs
