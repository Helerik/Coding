#!/usr/bin/env python3

# Vectorized implementation of logistic regression

import numpy as np

class LogisticRegressor():

    def __init__(self, trainX, trainY):
        self.X = trainX
        self.y = trainY
        self.m, self.n = trainX.shape
        if self.m != len(trainY):
            raise ValueError("Invalid vector sizes for trainX and trainY -> trainX size = " + self.m + " while trainY size = " + len(trainY))
        self.w = np.random.uniform(-1,1,n)
        self.b = 0

    # Sigmoid activation function
    def sigmoid(t):
        return 1/(1+np.exp(-t))

    # Loss function for  logistic regression
    def lossFunc(y, y_pred):
        return -(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))

    # Gradient descent for logistic regression
    def gradientDescent(self, b, alpha, max_iter):
        
        for _ in range(max_iter):
            
            Z = np.dot(self.w.T, self.X) + b
            y_pred = sigmoid(Z)
            costFunc = np.sum(lossFunc(self.y, y_pred))/self.m

            dz = y_pred - self.y
            dw = np.dot(self.X, dz.T)/self.m
            db = np.sum(dz)/self.m

            self.w -= alpha*dw
            b -= alpha*db

        self.b = b

        return [w, b]
