#!/usr/bin/env python3

import numpy as np

def sigmoid(t):
    return 1/(1+np.exp(-t))

def logisticRegression(x, w, b):
    return sigmoid(np.dot(w.T, x) + b)

def lossFunc(y, y_t):
    return -(y*np.log(y_t) + (1-y)*np.log(1-y_t))

def costFunc(w, b):
    for i in range(len())
    
