#!/usr/bin/env python3
# Author: Erik Davino Vincent

import numpy as np
from ActivationFunction import *

class Layer():

    def __init__(self, size, learning_rate, L2, activation, is_output = False):
        
        self.size = size
        self.learning_rate = learning_rate
        self.L2 = L2

        if activation.lower() == 'sigmoid':
            activation = Sigmoid
        elif activation.lower() == 'tanh':
            activation = Tanh
        elif activation.lower() == 'ltanh':
            activation = LeakyTanh
        elif activation.lower() == 'relu':
            activation = ReLu
        elif activation.lower() == 'softmax':
            activation = Softmax

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

    def backward_pass(self, A_prev, dA, Y = None):

        m = A_prev.shape[1]

        Z = self.Z

        W = self.W
        b = self.b

        if is_output:
            dZ = self.A - Y
        else:
            dZ = dA * self.activation.derivative(Z)

        dA_prev = np.dot(W.T, dZ)

        dW = (np.dot(A_prev, dZ.T) + self.L2*W.T)/m
        db = np.sum(dZ, axis = 1, keepdims = 1)/m

        W = W - self.learning_rate*W.T
        b = b - self.learning_rate*b
        self.W = W
        self.b = b









    

        
