#!/usr/bin/env python3
# Author: Erik Davino Vincent

import sys
sys.path.insert(1, 'C:/Users/Cliente/Desktop/Coding/Python Codes/Multilayered Neural Network')

import numpy as np
import matplotlib.pyplot as plt

from NN_Layers import *
from Metrics import *

from sklearn.model_selection import train_test_split
from mnist import MNIST

def prepare_MNIST_data():
    
    # import MNIST dataset
    mndata = MNIST('C:\\Users\\Cliente\\Desktop\\Coding\\Python Codes\\Multilayered Neural Network\MNIST')
    X, y = mndata.load_training()
    X = np.asarray(X)
    
    y_tmp = []
    for i in range(len(y)):
        app = [0 for _ in range(10)]
        app[y[i]] = 1
        y_tmp.append(app)
    y = np.array(y_tmp)

    # Scales X to (0,1) range
    X = X/255

    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size = 0.10)
    
    X_train = X_train.T
    X_dev = X_dev.T
    y_train = y_train.T
    y_dev = y_dev.T

    return (X_train, X_dev, y_train, y_dev)

def cost_function(y_pred, y_true):
    
    loss_func = -np.sum(y_true*np.log(y_pred), axis = 0, keepdims = 1)
    cost_func = np.mean(loss_func)
    
    return cost_func

def make_minibatches(X, Y, minibatch_size):

    m = X.shape[1]
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    minibatches = []

    num_full_minibatches = int(np.floor(m/minibatch_size))
    for k in range(num_full_minibatches):
        minibatch_X = shuffled_X[:, minibatch_size*k : minibatch_size*(k+1)]
        minibatch_Y = shuffled_Y[:, minibatch_size*k : minibatch_size*(k+1)]
        minibatch = (minibatch_X, minibatch_Y)
        minibatches.append(minibatch)

    # Depending on size of X and size of minibatches, the last minibatch will be smaller
    if m % minibatch_size != 0:
        minibatch_X = shuffled_X[:, minibatch_size*num_full_minibatches :]
        minibatch_Y = shuffled_Y[:, minibatch_size*num_full_minibatches :]
        minibatch = (minibatch_X, minibatch_Y)
        minibatches.append(minibatch) 
    
    return minibatches
    
def model(X, y, max_iter = 100, learning_rate = 0.001, activation = 'relu', minibatch_size = 1024):

    l1 = Layer(
        size = 20,
        activation = activation,
        optimizer = Adam(learning_rate = learning_rate))
    l2 = Layer(
        size = 20,
        activation = activation,
        optimizer = Adam(learning_rate = learning_rate))
    l3 = Layer(
        size = y.shape[0],
        activation = 'softmax',
        optimizer = Adam(learning_rate = learning_rate),
        is_output = True)
    
    l1.init_weights(X.shape[0])
    l2.init_weights(l1.size)
    l3.init_weights(l2.size)

    costs = []
    iters = []

    # Learning loop
    k = 0
    for it in range(max_iter):

        minibatches = make_minibatches(X, y, minibatch_size)

        for minibatch in minibatches:

            X_mb, y_mb = minibatch
        
            # Forward propagation
            l1.forward_pass(X_mb)
            l2.forward_pass(l1.A)
            l3.forward_pass(l2.A)

            # Evaluate and plot cost
            cost = cost_function(l3.A, y_mb)
            costs.append(cost)
            if k > 200:
                costs.pop(0)
                iters.pop(0)
            iters.append(k)
            k += 1
            plt.clf()
            plt.plot(iters, costs, color = 'b')
            plt.xlabel(f"Iteration {k}")
            plt.ylabel(f"Cost of {cost:.3}")
            plt.title(f"Cost for {it} epochs")
            plt.pause(0.001)

            # Backward propagation
            l3.backward_pass(l2.A, Y = y_mb)
            l2.backward_pass(l1.A, dA = l3.dA_prev)
            l1.backward_pass(X_mb, dA = l2.dA_prev)

    return (l1, l2, l3)


def main():

    X_train, X_dev, y_train, y_dev = prepare_MNIST_data()

    l1, l2, l3 = model(X_train, y_train, learning_rate = 0.005, max_iter = 10, minibatch_size = 1024)

    # Make prediction for X_train
    l1.forward_pass(X_train)
    l2.forward_pass(l1.A)
    l3.forward_pass(l2.A)
    y_pred = l3.A
    prediction = []
    truth = []
    for i in range(y_pred.shape[1]):
        prediction.append(np.argmax(y_pred[:,i]))
        truth.append(np.argmax(y_train[:,i]))
    y_pred =  np.array([prediction])
    y_train = np.array([truth])
    
    print()
    print(Metrics.score_table(y_train, y_pred))
    print()

    # Make prediction for X_dev
    l1.forward_pass(X_dev)
    l2.forward_pass(l1.A)
    l3.forward_pass(l2.A)
    y_pred = l3.A
    prediction = []
    truth = []
    for i in range(y_pred.shape[1]):
        prediction.append(np.argmax(y_pred[:,i]))
        truth.append(np.argmax(y_dev[:,i]))
    y_pred =  np.array([prediction])
    y_dev = np.array([truth])

    print()
    print(Metrics.score_table(y_dev, y_pred))
    print()

main()

















