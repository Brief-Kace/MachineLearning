from typing import Any, Union

import numpy


class node:
    inputs_array = []
    weights_array = []
    old_weights_array = []
    output = 0
    activate_func = 'logistic'
    bias = -1
    bias_weight = numpy.random.rand(1)
    error = 0
    node_number = -1

    def __init__(self, previous_layer, node_number):
        self.weights_array = numpy.random.rand(len(previous_layer))
        self.weights_array = self.weights_array*2 # range them from 0 to 2
        self.weights_array = self.weights_array-1 # range them from -1 to 1
        self.updated_weights_array = [0] * len(self.weights_array)
        self.inputs_array = previous_layer
        self.node_number = node_number

    def reset_weights(self):
        self.weights_array = numpy.random.rand(len(self.weights_array))

    def update_weights(self, learning_rate):
        #print(self.error, "\n", self.output)
        #print(self.weights_array - self.error*self.output*learning_rate)
        self.old_weights_array = self.weights_array
        self.weights_array = self.weights_array - (self.error * self.output * learning_rate)
        self.error = 0
        #print(self.weights_array- self.old_weights_array)

    def update_inputs(self, new_inputs):
        self.inputs_array = new_inputs

    def update_error(self, next_layer_array, target):
        error = 0
        if target:
            error = self.calc_error(next_layer_array=[], target=target)
        else:
            for i in range(0, len(next_layer_array)):
                error += self.calc_error(next_layer_array=next_layer_array[i], target=0)
        print("update_error")

        self.error += error

    def calc_error(self, next_layer_array, target):
        error = 0
        if next_layer_array:
            error = self.output_error(target)
        else:
            error = self.hidden_error(next_layer_array)
        print("calc_error")
        return error

    def output_error(self, target):
        return self.output*(1-self.output)*(self.output - target)

    def hidden_error(self, next_layer_array):
        x = 0
        for i in range(len(next_layer_array)):
            x += next_layer_array[i].weights_array[self.node_number] * next_layer_array[i].error
        return self.output * (1 - self.output) * x

    def update_output(self):
        x = []
        # if isinstance(self.inputs_array[0], node):
        #print(type(self.inputs_array))
        if type(self.inputs_array[0]) is not numpy.float64:
            for item in self.inputs_array:
                x.append(item.output)
        else:
            x = self.inputs_array
        x = [a * b for a, b in zip(x, self.weights_array)]
        x = sum(x) + (self.bias * self.bias_weight)
        self.output = self.activation(x)

    def activation(self, x):
        if 'logistic' in self.activate_func:
            return 1/(1+numpy.exp(-x))
        else:
            return -1
