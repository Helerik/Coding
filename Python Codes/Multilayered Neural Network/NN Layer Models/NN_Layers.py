#!/usr/bin/env python3
# Author: Erik Davino Vincent

import sys
sys.path.insert(1, 'C:/Users/Cliente/Desktop/Coding/Python Codes/Multilayered Neural Network')

import numpy as np
from ActivationFunction import *

from sklearn.model_selection import train_test_split
from mnist import MNIST

from Metrics import *

class Layer():

    def __init__(self, size, learning_rate, L2, activation, is_output = False):
        
        self.size = size
        self.learning_rate = learning_rate
        self.L2 = L2

        self.is_output = is_output

        if activation.lower() == 'sigmoid':
            self.activation = Sigmoid
        elif activation.lower() == 'tanh':
            self.activation = Tanh
        elif activation.lower() == 'ltanh':
            self.activation = LeakyTanh
        elif activation.lower() == 'relu':
            self.activation = ReLu
        elif activation.lower() == 'softmax':
            self.activation = Softmax

        self.Z = None
        self.A = None
        self.dA_prev = None

        self.W = None
        self.b = None

    def init_weights(self, size_prev):

            self.W = np.random.randn(self.size, size_prev)*np.sqrt(2/size_prev)
            self.b = np.zeros((self.size, 1))    

    def forward_pass(self, A_prev):

        W = self.W
        b = self.b

        Z = np.dot(W, A_prev) + b
        A = self.activation.function(Z)

        self.Z = Z
        self.A = A

    def backward_pass(self, A_prev, dA = None, Y = None):

        m = A_prev.shape[1]

        Z = self.Z

        W = self.W
        b = self.b

        if self.is_output:
            dZ = self.A - Y
        else:
            dZ = dA * self.activation.derivative(Z)

        self.dA_prev = np.dot(W.T, dZ)

        dW = (np.dot(A_prev, dZ.T) + self.L2*W.T)/m
        db = np.sum(dZ, axis = 1, keepdims = 1)/m

        W = W - self.learning_rate*dW.T
        b = b - self.learning_rate*db
        self.W = W
        self.b = b

def model():

    mndata = MNIST('C:\\Users\\Cliente\\Desktop\\Coding\\Python Codes\\Multilayered Neural Network\MNIST')
    X_train, y_train = mndata.load_training()
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    X_train = X_train/255
    y_tmp = []
    num = np.max(y_train)
    for i in range(len(y_train)):
        y_tmp.append([0 for _ in range(num+1)])
        y_tmp[i][y_train[i]] = 1
    y_train = np.array(y_tmp)

    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size = 0.10)
    
    X_train = X_train.T
    X_dev = X_dev.T
    y_train = y_train.T
    y_dev = y_dev.T

    l1 = Layer(size = 15, learning_rate = 0.1, L2 = 0, activation = 'relu', is_output = False)
    l1.init_weights(X_train.shape[0])
    
    l2 = Layer(size = 15, learning_rate = 0.1, L2 = 0, activation = 'relu', is_output = False)
    l2.init_weights(l1.size)
    
    l3 = Layer(size = 10, learning_rate = 0.1, L2 = 0, activation = 'softmax', is_output = True)
    l3.init_weights(l2.size)

    for i in range(100):

        l1.forward_pass(X_train)
        l2.forward_pass(l1.A)
        l3.forward_pass(l2.A)

        loss = -np.sum(y_train*np.log(l3.A), axis = 0, keepdims = 1)
        cost = np.mean(loss)
        print(cost)

        l3.backward_pass(l2.A, dA = None, Y = y_train)
        l2.backward_pass(l1.A, dA = l3.dA_prev, Y = None)
        l1.backward_pass(X_train, dA = l2.dA_prev, Y = None)

model()







    







        
