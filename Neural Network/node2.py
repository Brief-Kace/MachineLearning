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
    layer_number = -1
    node_number = -1


    def __str__(self):
        return "Layer, Node: " + str(self.layer_number) + " " + str(self.node_number) + "\nWeights: " +str(self.weights_array) + "\nCurrent Error: " + str(self.error)



    def __init__(self, previous_layer, layer_number, node_number):
        self.weights_array = numpy.random.rand(len(previous_layer))
        self.layer_number = layer_number
        self.weights_array = self.weights_array*2 # range them from 0 to 2
        self.weights_array = self.weights_array-1 # range them from -1 to 1
        self.updated_weights_array = [0] * len(self.weights_array)
        self.inputs_array = previous_layer
        self.node_number = node_number

    def reset_weights(self):
        self.weights_array = numpy.random.rand(len(self.weights_array)-1)

    def update_weights(self, learning_rate):
        #print("Error: ", self.error, "\nOutput: ", self.output)
        #print(self.error * self.output * learning_rate)
        #print(self.weights_array - ([self.error * self.output * learning_rate]*len(self.weights_array)))
        #exit(0)
        self.old_weights_array = self.weights_array
        self.weights_array = self.weights_array - ([self.error * self.output * learning_rate]*len(self.weights_array))
        self.error = 0
        #print(len(self.weights_array))
        #print(self.weights_array - self.old_weights_array)
    def update_inputs(self, new_inputs):
        self.inputs_array = new_inputs
        #print(self.inputs_array)

    def update_error(self, next_layer_array, target):
        error = 0
        #print(type(target))
        if 'list' in str(type(target)):
            error = self.output_error(target=target)
        else:
            error += self.hidden_error(next_layer_array=next_layer_array)
        self.error += error

    def output_error(self, target):
        #print("Output Error ", self.output * (1 - self.output) * (self.output - target))
        return self.output*(1-self.output)*(self.output - target[self.node_number])

    def hidden_error(self, next_layer_array):
        x = 0
        #print("next layer size: ", len(next_layer_array[0].weights_array))
        for i in range(0, (len(next_layer_array))):
            #print("i: ", i)
            #print(self.node_number)
            #print(len(next_layer_array))
            #print(len(next_layer_array[i].weights_array))
            x += next_layer_array[i].weights_array[self.node_number] * next_layer_array[i].error
        #print("Hidden Error ", self.output * (1 - self.output) * x)
        return self.output * (1 - self.output) * x

    def update_output(self):
        x = []
        # if isinstance(self.inputs_array[0], node):
        #print(type(self.inputs_array))
        if type(self.inputs_array[0]) is not numpy.float64:
            for item in self.inputs_array:
                x.append(item.output)
        else:
            #print("else")
            x = self.inputs_array
        #sum_value = 0
        #print( x)
        #print( self.weights_array)
        sum_value = sum([a * b for a, b in zip(x, self.weights_array)])
        #for i in range(len(x)):
            #sum_value += self.weights_array[i] * x[i]
        sum_value += self.bias * self.bias_weight

        #print(self.activation(sum_value))
        #exit(0)

        self.output = self.activation(sum_value[0])

    def activation(self, x):
        if 'logistic' in self.activate_func:
            return 1/(1+numpy.exp(-x))
        else:
            return -1