#!/usr/bin/env python3
# Author: Erik Davino Vincent

import sys
sys.path.insert(1, 'C:/Users/Cliente/Desktop/Coding/Python Codes/Multilayered Neural Network')

import numpy as np

from sklearn.model_selection import train_test_split
from mnist import MNIST

from NeuralNetwork import *
from Metrics import *

def example():

    # import MNIST dataset
    mndata = MNIST('C:\\Users\\Cliente\\Desktop\\Coding\\Python Codes\\Multilayered Neural Network\MNIST')
    X_train, y_train = mndata.load_training()
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    # An example in the dataset
    idx = np.random.randint(low = 0, high = len(X_train)-1)
    plt.imshow(np.array(X_train[idx]).reshape(28,28), cmap = 'Greys')
    plt.colorbar()
    plt.title(f"Number {y_train[idx]} from MNIST dataset (without scalling)")
    plt.show()

    # Scalling X
    X_train = X_train/255

    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size = 0.10)
    
    X_train = X_train.T
    X_dev = X_dev.T
    y_train = np.array([y_train])
    y_dev = np.array([y_dev])

    # Initializes NN classifier
    clf = NeuralNetwork(
        layer_sizes = [20,20],
        learning_rate = 0.005,
        max_iter = 200,
        L2 = 0,
        beta1 = 0.9,
        beta2 = 0.999,
        minibatch_size = 1024,
        activation = 'relu',
        classification = 'multiclass',
        plot_N = 0,
        end_on_close = False,
        end_on_delete = False)

    print()
    print()
    print(clf)

    clf.fit(X_train, y_train)

    # Make predictions
    predicted_y = clf.predict(X_train)
    table = Metrics.score_table(y_train, predicted_y)
    print()
    print()
    print("    Results for train set")
    print(table)

    predicted_y = clf.predict(X_dev)
    table = Metrics.score_table(y_dev, predicted_y)
    print()
    print()
    print("    Results for dev set")
    print(table)

    # Import test set
    X_test, y_test = mndata.load_testing()
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    X_test = X_test/255

    X_test = X_test.T
    y_test = np.array([y_test])

    # Make predictions for test set
    predicted_y = clf.predict(X_test)
    table = Metrics.score_table(y_test, predicted_y)
    print()
    print()
    print("    Results for test set")
    print(table)

    # Plots a mislabeled test example
    plt.figure()
    indexes = np.where(predicted_y != y_test)[1]
    idx = np.random.choice(indexes)
    plt.imshow(X_test.T[idx].reshape(28,28), cmap = 'Greys')
    plt.colorbar()
    plt.title(f"Number {y_test[0,idx]}, mislabeled as {predicted_y[0,idx]}")
    plt.show()
    
example()
