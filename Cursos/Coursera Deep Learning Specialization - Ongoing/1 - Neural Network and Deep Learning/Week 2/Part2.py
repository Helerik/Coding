#!/usr/bin/env python3

# Vectorized implementation of logistic regression

import numpy as np

class LogisticRegressor():
    def __init__(self, trainX, trainY):

        # Training data
        self.X = trainX
        self.y = trainY

        # Training data size
        self.n, self.m = trainX.shape
        if self.m != len(trainY):
            raise ValueError("Invalid vector sizes for trainX and trainY -> trainX size = " + str(self.m) + " while trainY size = " + str(len(trainY)))

        # Parameter vector
        self.w = np.random.uniform(-1,1,self.n)
        self.b = 0

        self.isTrained = 0

    # Sigmoid activation function
    def sigmoid(self, t):
        warnings.filterwarnings("")
        return 1/(1+np.exp(-t))

    # Loss function for  logistic regression
    def lossFunc(self, y, y_pred):
        return -(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))

    # Gradient descent for logistic regression
    def gradientDescent(self, alpha, max_iter):

        X = self.X
        y = self.y
        w = self.w
        b = self.b
        m = self.m
        
        for _ in range(max_iter):
            
            Z = np.dot(w.T, X) + b
            y_pred = self.sigmoid(Z)
            costFunc = np.sum(self.lossFunc(self.y, y_pred))/m

            dz = y_pred - self.y
            dw = np.dot(self.X, dz.T)/m
            db = np.sum(dz)/m

            w -= alpha*dw
            b -= alpha*db

        self.X = X
        self.y = y
        self.w = w
        self.b = b

        self.isTrained = 1

def example():
    
    X = np.array([
        [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
        [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
        [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
        ])
    y = np.array([1,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1])

    LR = LogisticRegressor(X, y)

    LR.gradientDescent(0.01, 1000)
    print(LR.isTrained)



example()







    
