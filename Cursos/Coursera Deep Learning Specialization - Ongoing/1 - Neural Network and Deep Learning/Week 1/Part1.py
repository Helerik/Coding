#!/usr/bin/env python3

import numpy as np

class LogisticLearner():

    # Constructor
    def __init__(self, trainX, trainY):
        self.trainX = trainX
        self.trainY = trainY

    def sigmoid(t):
        return 1/(1+np.exp(-t))

    def logisticRegression(x, w, b):
        return sigmoid(np.dot(w.T, x) + b)

    def lossFunc(y, y_t):
        return -(y*np.log(y_t) + (1-y)*log(1-y_t))
    
