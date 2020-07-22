# !/usr/bin/env python3
# Author: Erik Davino Vincent

import sys
sys.path.insert(1, 'C:/Users/Cliente/Desktop/Coding/Python Codes/Multilayered Neural Network/')

import numpy as np

from ActivationFunction import *

class RRN_Layer():

    def __init__(self, size, activation, learning_rate, L2, weights = None, biases = None):

        self.size = size
        self.learning_rate = learning_rate
        self.L2=  L2

        if activation.lower() == 'sigmoid':
            self.activation = Sigmoid
        elif activation.lower() == 'tanh':
            self.activation = Tanh
        elif activation.lower() == 'ltanh':
            self.activation = LeakyTanh
        elif activation.lower() == 'relu':
            self.activation = ReLu

        self.X = None
        self.y = None
        
        self.Z_values = None
        self.activations = None
        
        self.dA = None
        self.dA_prev = None
        
        self.weights = weights
        self.biases = biases

    def init_weights(self, prevlayer):

        n_h_prev = prevlayer.size
        n_h = self.size

        self.weights = np.random.randn(n_h, n_h_prev)*np.sqrt(2/n_h_prev)
        self.biases = np.zeros((n_h,1))

    def forward_propapagtion(self, prevlayer):

        A_prev = prevlayer.activations
        X = self.X
        vector = np.concatenate((A_prev, X), axis = 1)
        
        W = self.weights
        b = self.biases

        Z = np.dot(W, vector) + b
        A = self.activation.function(Z)

        self.activations = A
        self.Z_values = Z

    def backward_propagation(self, prevlayer):

        
        A_prev = prevlayer.activations
        Z = self.Z_vals

        m = A_prev.shape[1]

        W = self.weights
        b = self.biases

        self.dA = nextlayer.dA_prev
        dZ = self.dA * self.activation.derivative(Z)
        self.dA_prev = np.dot(W.T, dZ)

        dW = np.dot(A_prev, dZ.T)/m + self.L2*W.T/m
        db = np.mean(dZi, axis = 1, keepdims = 1)

        W = W - self.learning_rate*dW.T
        b = b - self.learning_rate*db

        self.weights = W
        self.weights = b
 
        
        








