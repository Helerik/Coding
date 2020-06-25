#!/usr/bin/env python3

# Vectorized implementation of logistic regression

import numpy as np

def sigmoid(t):
    return 1/(1+np.exp(-t))

def lossFunc(y, y_pred):
    return -(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))

def gradientDescent(trainX, trainY, w, b, alpha, iters):
    
    m = len(trainX)
    if m != len(trainY):
        return False

    n = len(w)

    X = np.copy(trainX)
    y = np.copy(trainY)

    dw = np.zeros(n)
    db = 0
    costFunc = 0
    
    for it in range(iters):
        for i in range(m):
            
            Z = np.dot(w.T, X) + b
            y_pred = sigmoid(z)
            costFunc += lossFunc(y[i], y_pred)

            dz = y_pred - y
            db += dz
            dw += X[i] * dz

        costFunc /= m
        
        db /= m
        b -= alpha*db
        dw /= m
        w -= alpha*dw

    return [w, b]
