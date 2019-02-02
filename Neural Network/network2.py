from node2 import node
import numpy
from sklearn.metrics import confusion_matrix
from prettytable import PrettyTable


class network:
    hidden_layers = []
    epochs = 100

    input_layer = []
    output_layer = []

    learning_rate = 1
    number_of_classes = -1

    data_train = []
    targets_train = []

    confusion_matrix = -1

    data_test = -1
    targets_test = -1



    def __init__(self, num_epochs, hidden_layers, learning_rate, data_train, targets_train, data_test, targets_test):
        self.epochs = num_epochs
        self.data_train = data_train
        self.targets_train = targets_train
        self.learning_rate = learning_rate
        self.input_layer = [0] * len(data_train[0])

        for j in range(0, len(hidden_layers)):
            #print(j)
            list_of_nodes = []
            for i in range(0, hidden_layers[j]):
                if j == 0:
                    list_of_nodes.append(node(self.input_layer, j, i))
                else:
                    list_of_nodes.append(node(self.hidden_layers[j-1], j, i))
            self.hidden_layers.append(list_of_nodes)

        classes = numpy.unique(targets_train)
        self.learning_rate = learning_rate

        #print(classes)
        self.number_of_classes = len(classes)

        self.data_test = data_test
        self.targets_test = targets_test
        for i in range(0, self.number_of_classes):
            self.output_layer.append(node(self.hidden_layers[-1], "Output", i))

    def classify(self, input_data):
        self.change_input_layer(input_data)
        self.feed_forward()
        result = self.find_max()
        return result

    def classify_multiple(self, input_data):
        results = []
        for i in range(0, len(input_data)):
            results.append(self.classify(input_data[i]))
        return results

    def print_errors(self):
        for i in range(0, len(self.hidden_layers)):
            for j in range(0, len(self.hidden_layers[i])):
                print(self.hidden_layers[i][j])
        for i in range(0, len(self.output_layer)):
            print(self.output_layer[i])

    def change_input_layer(self, new_input_layer):
        for i in range(0, len(self.hidden_layers[0])):
            self.hidden_layers[0][i].update_inputs(new_input_layer)

    def feed_forward(self):
        for i in range(0, len(self.hidden_layers)):
            for j in range(0, len(self.hidden_layers[i])):
                self.hidden_layers[i][j].update_output()
        for i in range(0, len(self.output_layer)):
            self.output_layer[i].update_output()

    def back_propagate(self, target):
        for i in range( 0, len(self.output_layer)):
            #pass
            #print("Output_Layer: ")
            self.output_layer[i].update_error(target=target, next_layer_array=[])
        #print(len(self.hidden_layers))
        #print("Hidden Layers: ")
        for i in range(len(self.hidden_layers)-1, 0, -1):
            #print(self.hidden_layers[0])
            #exit(0)
            if i == len(self.hidden_layers)-1:
                #print(len(self.hidden_layers[i]))
                for j in range(len(self.hidden_layers[i])):
                    self.hidden_layers[i][j].update_error(next_layer_array=self.output_layer, target=-1)
            else:
                for j in range(len(self.hidden_layers[i])):
                    #print(type(self.hidden_layers[i]))
                    #print(i,j)
                    self.hidden_layers[i][j].update_error(next_layer_array=self.hidden_layers[i+1], target=-1)

    def find_max(self):
        outputs = []
        for i in range(0, len(self.output_layer)):
            outputs.append(self.output_layer[i].output)
        return outputs.index(max(outputs))

    def update_weights(self):
        for i in range(0, len(self.hidden_layers)):
            for j in range(0, len(self.hidden_layers[i])):
                #print("hidden_layers[", i, "][", j, "]")
                self.hidden_layers[i][j].update_weights(self.learning_rate)
        for i in range(0, len(self.output_layer)):
            self.output_layer[i].update_weights(self.learning_rate)

    def find_accuracy(self):
        accuracy_count = 0
        #print(self.number_of_classes)
        confusion_matrix = [[0] * self.number_of_classes] * self.number_of_classes
        #print(len(self.data_test))
        #print(len(self.targets_test))
        for i in range(0, (len(self.data_test))):
            self.classify(self.data_test[i])
            result = self.find_max()
            #print("i ", i)
            #print(len(self.targets_test))
            #print(self.targets_test[i])
            #print(len(confusion_matrix[self.targets_test[i]]))
            #print(self.targets_test[i], result)
            #print("Adding 1 to confusionmatrix [", self.targets_test[i], "][", result, "]")
            confusion_matrix[self.targets_test[i]][result] = 1 + confusion_matrix[self.targets_test[i]][result]
            if result == self.targets_test[i]:
                accuracy_count += 1
        self.confusion_matrix = confusion_matrix
        return accuracy_count/len(self.targets_test)

    def train(self):
        for i in range(0, self.epochs):
            self.single_epoch()


    def single_epoch(self):
        for i in range(0, len(self.data_train)):
            self.classify(self.data_train[i])
            #print(self.output_layer[])
            #print(self.find_max())
            target_output_layer = [0] * len(self.output_layer)
            target_output_layer[self.targets_train[i]] = 1
            self.back_propagate(target_output_layer)
        self.print_errors()

        self.update_weights()
        #print("end of first epoch")
        #exit(0)
        #print(self.confusion_matrix)
        classes_predicted = self.classify_multiple(self.data_test)

        # matrix = confusion_matrix(self.targets_test, classes_predicted)
        # t = PrettyTable(['', 'Copper Sheet', 'Tennis Racket', 'Brigham', 'Kace'])
        # t.add_row(['Copper Sheet', matrix[0, 0], matrix[0, 1], matrix[0, 2], matrix[0, 3]])
        # t.add_row(['Tennis Racket', matrix[1, 0], matrix[1, 1], matrix[1, 2], matrix[1, 3]])
        # t.add_row(['Brigham', matrix[2, 0], matrix[2, 1], matrix[2, 2], matrix[2, 3]])
        # t.add_row(['Kace', matrix[3, 0], matrix[3, 1], matrix[3, 2], matrix[3, 3]])
        # print(t)

        print("Accuracy: ", self.find_accuracy())




