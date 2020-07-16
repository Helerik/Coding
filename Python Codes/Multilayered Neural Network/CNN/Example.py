#!/usr/bin/env python3
# Author: Erik Davino Vincent

import sys
sys.path.insert(1, 'C:/Users/Cliente/Desktop/Coding/Python Codes/Multilayered Neural Network')

import numpy as np

from sklearn.model_selection import train_test_split
from mnist import MNIST

from CNN import *
from Metrics import *

def example():

    # import MNIST dataset
    mndata = MNIST('C:\\Users\\Cliente\\Desktop\\Coding\\Python Codes\\Multilayered Neural Network\MNIST')
    X_train, y_train = mndata.load_training()
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    X_train = X_train/255

    new_X = []
    for i in range(X_train.shape[0]):
        new_X.append(np.array(X_train[i]).reshape(1,28,28))
    X_train = np.array(new_X)

    # Will perform training on 1500 images from the dataset -10% for dev
    X_train = X_train[:1500]
    y_train = y_train[:1500]

    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size = 0.10)
    
    y_train = np.array([y_train])
    y_dev = np.array([y_dev])

    # Initializes NN classifier
    clf = CNN(
        layer_sizes = [
                     {'type':'conv', 'f_H':3, 'f_W':3, 'n_C':3, 'stride':1, 'pad':0},
                     {'type':'pool', 'f_H':2, 'f_W':2, 'n_C':6, 'stride':2, 'mode':'max'},
                     {'type':'fc', 'size':10}
                     ],
        learning_rate = 0.0005,
        max_iter = 75,
        L2 = 1, # Not working yet
        beta1 = 0.9, # Not working yet
        beta2 = 0.999, # Not working yet
        minibatch_size = 100,
        activation = 'relu',
        classification = 'multiclass',
        plot_N = 1,
        end_on_close = True,
        end_on_backspace = True)

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

    new_X = []
    for i in range(X_test.shape[0]):
        new_X.append(np.array(X_test[i]).reshape(1,28,28))
    X_test = np.array(new_X)

    y_test = np.array([y_test])

    # Make predictions for test set
    predicted_y = clf.predict(X_test)
    table = Metrics.score_table(y_test, predicted_y)
    print()
    print()
    print("    Results for test set")
    print(table)
    
example()
