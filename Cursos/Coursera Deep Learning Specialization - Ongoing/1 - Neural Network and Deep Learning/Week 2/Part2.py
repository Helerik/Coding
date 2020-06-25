#!/usr/bin/env python3

# Vectorized implementation of logistic regression

import numpy as np

# Sigmoid activation function
def sigmoid(t):
    return 1/(1+np.exp(-t))

# Loss function for  logistic regression
def lossFunc(y, y_pred):
    return -(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))

# Gradient descent for logistic regression
def gradientDescent(trainX, trainY, w, b, alpha, iters):
    
    m = len(trainX)
    if m != len(trainY):
        raise ValueError("Invalid vector sizes for trainX and trainY -> trainX size = " + m + " while trainY size = " + len(y))
    n = len(w)

    X = np.copy(trainX)
    y = np.copy(trainY)
    
    for it in range(iters):
        
        Z = np.dot(w.T, X) + b
        y_pred = sigmoid(z)
        costFunc = np.sum(lossFunc(y, y_pred))/m

        dz = y_pred - y
        dw = np.dot(X, dz.T)/m
        db = np.sum(dz)/m

        w -= alpha*dw
        b -= alpha*db

    return [w, b]
