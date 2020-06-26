#!/usr/bin/env python3

# Vectorized implementation of logistic regression

import numpy as np
import matplotlib.pyplot as plt

def model(trainX, trainY, alpha, max_iter):

    # Training data size
    n, m = trainX.shape
    if m != trainY.shape[1]:
        raise ValueError("Invalid vector sizes for trainX and trainY -> trainX size = " + str(self.m) + " while trainY size = " + str(len(trainY)))

    # Parameter vector
    w = np.zeros((n, 1))
    b = 0

    return gradientDescent(trainX, trainY, w, b, alpha, max_iter)

# Sigmoid activation function
def sigmoid(t):
    return 1/(1+np.exp(-t))

# Gradient descent for logistic regression
def gradientDescent(X, y, w, b, alpha, max_iter):

    cost = []
    m = X.shape[1]
    for _ in range(max_iter):

        y_pred = sigmoid(np.dot(w.T, X) + b)
        costFunc = np.sum(-(y*np.log(y_pred) + (1-y)*np.log(1-y_pred)))/m
        cost.append(costFunc)

        dz = y_pred - y
        dw = np.dot(X, dz.T)/m
        db = np.sum(dz)/m

        w = w - alpha*dw
        b = b - alpha*db

    plt.plot(cost)
    plt.title("Gradient Descent")
    plt.show(block = False)

    return [w, b]

# Predicts if X vector is 1 or 0
def predict(X, w, b):
    
    S = sigmoid(np.dot(w.T, X) + b)
    prediction = np.zeros((1, X.shape[1]))
    for i in range(S.shape[1]):
        if S[0,i] > 0.5:
            prediction[0, i] = 1
            
    return prediction

def example():
    
    X = np.array([
        [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
        [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
        [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
        ])
    y = np.array([[1,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]])
    X[0] = X[0]/100

    w, b = model(X, y, 0.05, 100000)

    pred = predict(X, w, b)

    percnt = 0
    for i in range(pred.shape[1]):
        if pred[0,i] == y[0,i]:
            percnt += 1
    percnt /= pred.shape[1]

    print("Accuracy of", percnt*100 , "%")

    

    

example()







    
