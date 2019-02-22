from network2 import network
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
3
from sklearn.preprocessing import normalize
import pandas
import numpy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from prettytable import PrettyTable


def main():
    data_brigham = pandas.read_csv("Brigham.csv")
    data_kace = pandas.read_csv("Kace.csv")
    data_copper_sheet = pandas.read_csv("Copper_Sheet.csv")
    data_tennis_racket = pandas.read_csv("Tennis_Racket.csv")

    classes = []
    for i in range(len(data_brigham)):
        classes.append(0)
    for i in range(len(data_kace)):
        classes.append(1)
    for i in range(len(data_tennis_racket)):
        classes.append(2)
    for i in range(len(data_copper_sheet)):
        classes.append(3)
    data = [
        data_copper_sheet.values,
        data_tennis_racket.values,
        data_brigham.values,
        data_kace.values
    ]
    data = numpy.concatenate(data)

    data = data[:, 1:]
    data = normalize(data)
    #iris = np.genfromtxt('iris.data', delimiter=',', dtype=None)


    #print(type(iris))
    #print(iris)
    #print(iris)
    #print (iris)
    #print(iris[:][-1])
    #iris_target = iris[:][-1]
    #iris_data = iris[:][1:]
    #print(iris_target)

    data_train, data_test, targets_train, targets_test = train_test_split( data, classes, test_size=0.6,
                                                                                  random_state=56)
    classes_test=targets_test
    #print(data_train[0])
    new_network = network(200, [20, 20, 20], 1, data_train, targets_train, data_test, targets_test, 56)
    new_network.train()
    classes_predicted = new_network.classify_multiple(data_test)

    matrix = confusion_matrix(classes_test, classes_predicted)
    t = PrettyTable(['','Copper Sheet', 'Tennis Racket', 'Brigham', 'Kace'])
    t.add_row(['Copper Sheet',  matrix[0,0], matrix[0,1], matrix[0,2], matrix[0,3]])
    t.add_row(['Tennis Racket', matrix[1,0], matrix[1,1], matrix[1,2], matrix[1,3]])
    t.add_row(['Brigham',       matrix[2,0], matrix[2,1], matrix[2,2], matrix[2,3]])
    t.add_row(['Kace',          matrix[3,0], matrix[3,1], matrix[3,2], matrix[3,3]])
    print(t)


main()


