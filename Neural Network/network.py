from node import node
import numpy

class network:
    hidden_layers = []
    epochs = 100

    input_layer = [5, 5, 5, 5, 5]
    output_layer = []

    learning_rate = 1
    epoch_limit = 50
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
        for j in range(0, len(hidden_layers)):
            list_of_nodes = []
            for i in range(0, hidden_layers[j]):
                if j == 0:
                    list_of_nodes.append(node(self.input_layer, i))
                else:
                    list_of_nodes.append(node(self.hidden_layers[j-1], i))
                self.hidden_layers.append(list_of_nodes)
        classes = numpy.unique(targets_train)
        self.learning_rate = learning_rate
        self.input_layer = [0] * len(data_train)
        #print(classes)
        self.number_of_classes = len(classes)

        self.data_test = data_test
        self.targets_test = targets_test
        for i in range(0, self.number_of_classes):
            self.output_layer.append(node(self.hidden_layers[-1], i))

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

    def change_input_layer(self, new_input_layer):
        for i in range(0, len(self.hidden_layers[0])):
            self.hidden_layers[0][i].update_inputs(new_input_layer)

    def feed_forward(self):
        for i in range(0, len(self.hidden_layers)):
            for j in range(0, len(self.hidden_layers[i])):
                self.hidden_layers[i][j].update_output()
        for i in range(0, len(self.output_layer)):
            self.output_layer[i].update_output()

    def back_propagate_multiple(self, targets):
        for i in range(len(targets)):
            print("Back_Propagate_Multiple")
            target_output_layer = [0]*len(self.output_layer)
            target_output_layer[targets[i]] = 1
            self.back_propagate(target_output_layer)

    def back_propagate(self, target):
        print(len(self.hidden_layers[0]))
        for i in range( 0, len(self.output_layer)):
            self.output_layer[i].update_error(target=target, next_layer_array=[])
        for i in range(len(self.hidden_layers), 0, -1):
            self.hidden_layers[i].update_error(next_layer_array=self.hidden_layers[i+1], target=-1)


    def find_max(self):
        outputs = []
        for i in range(0, len(self.output_layer)):
            outputs.append(self.output_layer[i].output)
        return outputs.index(max(outputs))

    def update_weights(self):
        for i in range(0, len(self.hidden_layers)):
            for j in range(0, len(self.hidden_layers[i])):
                self.hidden_layers[i][j].update_weights(self.learning_rate)
        for i in range(0, len(self.output_layer)):
            self.output_layer[i].update_weights(self.learning_rate)

    def calculate_errors(self):
        for i in range(0, len(self.output_layer)):
            for i in range(0, len(self.output_layer[i].updated_weights_array)):
                self.output_layer[i].updated_weights_array += self.weight_difference()
        for i in range(0, len(self.hidden_layers)):
            for j in range(0, len(self.hidden_layers[i])):
                self.hidden_layers[i][j].updated_weights_array += self.weight_difference()

    def find_accuracy(self):
        accuracy_count = 0
        confusion_matrix = [[0] * self.number_of_classes] * self.number_of_classes
        for i in range(0, len(self.data_test)):
            self.classify(self.data_test[i])
            result = self.find_max()
            confusion_matrix[self.targets_train[i]][result] = 1 + confusion_matrix[self.targets_train[i]][result]
            if result == self.targets_train[i]:
                accuracy_count += 1
        self.confusion_matrix = confusion_matrix
        return accuracy_count/len(self.data_train)


    def train(self):
        for i in range(0, self.epoch_limit):
            self.single_epoch()


    def single_epoch(self):
        for i in range(0, len(self.data_train)):
            self.classify(self.data_train[i])
            #print(self.output_layer[])
            #print(self.find_max())
            target_output_layer = [0] * len(self.output_layer)
            target_output_layer[self.targets_train[i]] = 1
            self.back_propagate(target_output_layer)
        self.update_weights()
        print("end of first epoch")
        exit(0)

        print("Accuracy: ", self.find_accuracy())




