#!/usr/bin/env python3

# Vectorized implementation of logistic regression

import numpy as np
import matplotlib.pyplot as plt

def model(X, Y, layerSize, alpha, max_iter = 100, tol = 1e-20):

    n_x, m = X.shape
    n_h = layerSize
    n_y = Y.shape[0]
    if m != Y.shape[1]:
        raise ValueError("Invalid vector sizes for trainX and trainY -> trainX size = " + str(self.m) + " while trainY size = " + str(len(Y)))

    weights = initializeWeights(n_x, n_h, n_y)

    return mixedMethod(X, Y, weights, alpha, max_iter, tol)

# Sigmoid activation function
def sigmoid(t):
    return 1/(1+np.exp(-t))

# Initialize weights
def initializeWeights(n_x, n_h, n_y, seed = None):
    np.random.seed(seed)
    
    W1 = np.random.random((n_h, n_x))/10
    b1 = np.zeros((n_h,1))
    W2 = np.random.random((n_y, n_h))/10
    b2 = np.zeros((n_y,1))

    weights = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
        }
    
    np.random.seed(None)
    return weights

# Creates a positive definite aproximation to a not positive definite matrix
def aprox_pos_def(A):
    u, V = np.linalg.eig(A)
    U = np.diag(u)
    for i in range(len(U)):
        for j in range(len(U)):
            if U[i][j] < 0:
                U[i][j] = -U[i][j]
    B = np.dot(V, np.dot(U, V.T))
    
    return B

# Mixed method for logistic regression
def mixedMethod(X, Y, weights, alpha, max_iter, tol):

    W1 = weights['W1']
    b1 = weights['b1']
    W2 = weights['W2']
    b2 = weights['b2']

    cost = []
    iteration = []
    
    true_alpha = alpha
    for _ in range(max_iter):
        alpha = true_alpha

        # Forward
        Z1 = np.dot(W1, X) + b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2) # old y_pred

        # Evaluate cost function
        if _%10 == 0:
            lossFunc = -(Y*np.log(A2) + (1-Y)*np.log(1-A2))
            costFunc = np.mean(lossFunc)
            cost.append(costFunc)
            iteration.append(_)

        # "Backward"
        
        # Layer 2:
        dZ2 = A2 - Y

        m = A1.shape[1]

        gradVect2 = np.dot(A1, dZ2.T)/m # old dW2
        gradb2 = np.sum(dZ2, axis = 1, keepdims = 1)/m
        gradVect2 = np.append(gradVect2, gradb2, axis = 0)
        
        hessMatx2 = np.dot(A2, (1-A2).T) * np.dot(A1, A1.T)/m
        hessb2par = (np.dot(A2, (1-A2).T) * np.dot(A1, np.ones((A1.shape[1],1)))/m)
        hessb2 = (np.dot(A2, (1-A2).T)/m)
        hessMatx2 = np.concatenate((hessMatx2, hessb2par), axis = 1)
        hessb2par = np.concatenate((hessb2par, hessb2), axis = 0)
        hessMatx2 = np.concatenate((hessMatx2, hessb2par.T), axis = 0)

        hessMatx2 = aprox_pos_def(hessMatx2)
        delta2 = np.linalg.solve(hessMatx2, gradVect2)
            
        dW2 = delta2[:-1]
        db2 = delta2[-1]

        # Layer 1:
        dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)

        m = X.shape[1]
        
        gradVect1 = np.dot(X, dZ1.T)/m # old dW1
        gradb1 = np.sum(dZ1, axis = 1, keepdims = 1)/m
        gradVect1 = np.append(gradVect1, gradb1.T, axis = 0)

        for i in range(len(A1)):
            hessMatx1 = np.dot(np.array([A1[i]]), (1-np.array([A1[i]])).T) * np.dot(X, X.T)/m
            hessb1par = (np.dot(np.array([A1[i]]), (1-np.array([A1[i]])).T) * np.dot(X, np.ones((X.shape[1],1)))/m)
            hessb1 = np.dot(np.array([A1[i]]), (1-np.array([A1[i]])).T)/m
            hessMatx1 = np.concatenate((hessMatx1, hessb1par), axis = 1)
            hessb1par = np.concatenate((hessb1par, hessb1), axis = 0)
            hessMatx1 = np.concatenate((hessMatx1, hessb1par.T), axis = 0)
            hessMatx1 = aprox_pos_def(hessMatx1)
            if i == 0:
                delta1 = np.linalg.solve(hessMatx1, np.array([gradVect1[:,i]]).T).T
            else:
                delta1 = np.append(delta1, np.linalg.solve(hessMatx1, np.array([gradVect1[:,i]]).T).T, axis = 0)

        dW1 = delta1[:,:-1]
        db1 = np.array([delta1[:,-1]]).T

        print(dW2)
        print(dW1)

        # Tol check
        if np.max(np.linalg.norm(dW1, axis = 1, keepdims = 1)) < tol \
        or np.max(np.linalg.norm(db1, axis = 1, keepdims = 1)) < tol \
        or np.max(dW2) < tol \
        or np.max(db2) < tol:
            break

        # True update
        W1 = W1 - alpha*dW1
        b1 = b1 - alpha*db1
        W2 = W2 - alpha*dW2.T
        b2 = b2 - alpha*db2

    lossFunc = -(Y*np.log(A2) + (1-Y)*np.log(1-A2))
    costFunc = np.mean(lossFunc)
    cost.append(costFunc)
    iteration.append(_)

    start = iteration[-1]

    plt.clf()
    plt.plot(iteration, cost, color = 'b')
    iteration = []
    cost = []

    for _ in range(start, max_iter*2):

        lossFunc = -(Y*np.log(A2) + (1-Y)*np.log(1-A2))
        costFunc = np.mean(lossFunc)
        cost.append(costFunc)
        iteration.append(_)

        # Forward propagation
        Z1 = np.dot(W1, X) + b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)

        # Backward propagation
        dZ2 = A2 - Y
        m = A1.shape[1]
        dW2 = np.dot(dZ2, A1.T)/m
        db2 = np.sum(dZ2, axis = 1, keepdims = 1)/m
        
        dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
        m = X.shape[1]
        dW1 = np.dot(dZ1, X.T)/m
        db1 = np.sum(dZ1, axis = 1, keepdims = 1)/m

        W1 = W1 - alpha*dW1*0.001
        b1 = b1 - alpha*db1*0.001
        W2 = W2 - alpha*dW2*0.001
        b2 = b2 - alpha*db2*0.001

    weights = {
    "W1": W1,
    "b1": b1,
    "W2": W2,
    "b2": b2
    }

    plt.plot(iteration, cost, color = 'r')
    plt.title("Mixed Descent for Cost Function")
    plt.show(block = 0)

    return weights

# Predicts if X vector is 1 or 0
def predict(weights, X):

    W1 = weights['W1']
    b1 = weights['b1']
    W2 = weights['W2']
    b2 = weights['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    return (A2 > 0.5)

def example():
    
    X = np.array([
        [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
        [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
        [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
        ])
    y = np.array([[1,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]])
    X[0] = X[0]/100

    weights = model(X, y, 20, 0.0001, 10000, tol = 1e-20)

    pred = predict(weights, X)

    percnt = 0
    for i in range(pred.shape[1]):
        if pred[0,i] == y[0,i]:
            percnt += 1
    percnt /= pred.shape[1]

    print("Accuracy of", percnt*100 , "%")
example()
