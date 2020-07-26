# !/usr/bin/env python3
# Author: Erik Davino Vincent

import sys
sys.path.insert(1, 'C:/Users/Cliente/Desktop/Coding/Python Codes/Multilayered Neural Network')
sys.path.insert(1, 'C:/Users/Cliente/Desktop/Coding/Python Codes/Multilayered Neural Network/NN Layer Models')

import numpy as np
from ActivationFunction import *
from NN_Layers import *

class RNN_Cell():

    def __init__(self, output_size, hidden_size, optimizer, bptt_truncate = 4):

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.optimizer = optimizer
        self.bptt_truncate = bptt_truncate

        self.U = np.random.randn(hidden_size, output_size)*np.sqrt(2/output_size)
        self.V = np.random.randn(output_size, hidden_size)*np.sqrt(2/hidden_size)
        self.W = np.random.randn(hidden_size, hidden_size)*np.sqrt(2/hidden_size)
        self.bw = np.zeros((hidden_size, 1))
        self.bv = np.zeros((output_size, 1))

        self.A = None
        self.Y_pred = None

    def forward_pass(self, X):

        T_x = X.shape[2]

        A = np.zeros((self.hidden_size, T_x + 1))
        A[:,-1] = np.zeros((self.hidden_size, 1))

        Y_pred = np.zeros((self.input_dim, T_x + 1))

        for t in range(T_x):

            A[:,t] = Tanh(np.dot(self.U, X[:,t]) + np.dot(self.W, A[:,t-1]) + bw)
            Y_pred[:,t] = Softmax(np.dot(self.V, A[:,t]) + bv)

        self.A = A
        self.Y_pred = Y_pred

    def predict(self, X):
        return np.argmax(self.Y_pred, axis = 0)

    def backward_pass(self, X, Y):

        T_x = X.shape[2]

        A = self.A
        Y_pred = self.Y_pred

        dU = np.zeros(self.U.shape)
        dV = np.zeros(self.V.shape)
        dW = np.zeros(self.W.shape)
        dbw = np.zeros(self.bw.shape)
        dbv = np.zeros(self.bv.shape)

        dZy = Y_pred - Y

        for t in range(T_x):

            dV += np.dot(A[:,t], dZy[:,t].T)
            dbv += np.sum(dZy[:,t], axis = 1, keepdims = 1)
            
            dZ = np.dot(self.V.T, dZy[t]) * (1 - np.square(A[t]))

            for bp_step in np.arange(max(0, t - self.bptt_truncate), t+1)[::-1]:

                dW += np.dot(A[:,bp_step-1], dZ.T)
                dU += np.dot(X[:,bp_step], dZ.T)
                dbw += np.sum(dZ, axis = 1, keepdims = 1)

                dZ = np.dot(self.W.T, dZ) * (1 - np.square(A[bp_step-1]))

        V, bv = self.optimizer.update(np.array([[]]).T, dV, dbv, V, bv)
        W, bw = self.optimizer.update(np.array([[]]).T, dW, dbw, W, bw)
        U, _  = self.optimizer.update(np.array([[]]).T, dU, dbw, U, bw)














