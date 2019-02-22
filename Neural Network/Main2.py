from network2 import network
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import normalize



def main():
    iris = datasets.load_iris()
    #iris = np.genfromtxt('iris.data', delimiter=',', dtype=None)

    #print(type(iris))
    #print(iris)
    #print(iris)
    #print (iris)
    #print(iris[:][-1])
    #iris_target = iris[:][-1]
    #iris_data = iris[:][1:]
    #print(iris_target)
    normalize(iris.data)

    data_train, data_test, targets_train, targets_test = train_test_split( iris.data, iris.target, test_size=0.3,
                                                                                  random_state=56)
    #print(data_train[0])
    new_network = network(1000, [ 3, 3], 1, data_train, targets_train, data_test, targets_test, 56)
    new_network.classify_multiple(data_test)
    new_network.train()



main()
