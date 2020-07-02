#!/usr/bin/env python3
# Author: Erik Davino Vincent

# Almost pure Newton-Raphson method implementation of a neural network

import numpy as np
import matplotlib.pyplot as plt

# Fits for X and Y
def model(X, Y, layerSizes, learningRate, max_iter = 100, plotN = 100):

    n_x, m = X.shape
    n_y = Y.shape[0]
    if m != Y.shape[1]:
        raise ValueError("Invalid vector sizes for X and Y -> X size = " + str(X.shape) + " while Y size = " + str(Y.shape) + ".")

    weights = initializeWeights(n_x, n_y, layerSizes)
    numLayers = len(layerSizes)

    return newtonMethod(X, Y, weights, learningRate, numLayers, max_iter, plotN)

# Sigmoid activation function
def sigmoid(t):
    return 1/(1+np.exp(-t))

# Initializes weights for each layer
def initializeWeights(n_x, n_y, layerSizes, scaler = 0.1, seed = None):
    
    np.random.seed(seed)
    
    weights = {}
    n_hPrev = n_x
    for i in range(len(layerSizes)):
        n_h = layerSizes[i]
        weights['W'+str(i+1)] = np.random.randn(n_h, n_hPrev)*scaler
        weights['b'+str(i+1)] = np.zeros((n_h,1))
        n_hPrev = n_h
    
    weights['W'+str(len(layerSizes)+1)] = np.random.random((n_y, n_hPrev))*scaler
    weights['b'+str(len(layerSizes)+1)] = np.zeros((n_y,1))

    np.random.seed(None)
    
    return weights

# Creates a positive definite aproximation to a not positive definite matrix
def aprox_pos_def(A):

    u, V = np.linalg.eig(A)
    U = np.absolute(np.diag(u))
    B = np.dot(V, np.dot(U, V.T))
 
    return B

# Performs foward propagation
def forwardPropagation(weights, X, numLayers):

    # Cache for A
    Avals = {}
    AiPrev = np.copy(X)
    for i in range(numLayers+1):
        
        Wi = weights['W'+str(i+1)]
        bi = weights['b'+str(i+1)]
        
        Zi = np.dot(Wi, AiPrev) + bi
        Ai = sigmoid(Zi)

        Avals['A'+str(i+1)] = Ai
        AiPrev = np.copy(Ai)

    return Avals
    

# Newton method for training a neural network
def newtonMethod(X, Y, weights, learningRate, numLayers, max_iter, plotN):
    
    # Cache for ploting cost
    cost = []
    iteration = []

    # Cache for minimum cost and best weights
    bestWeights = weights.copy()
    minCost = np.inf

    # Init break_code = 0
    break_code = 0
    for it in range(max_iter):

        # Forward propagation
        Avals = forwardPropagation(weights, X, numLayers)

        # Evaluates cost fucntion
        AL = np.copy(Avals['A'+str(numLayers+1)])
        lossFunc = -(Y*np.log(AL) + (1-Y)*np.log(1-AL))
        costFunc = np.mean(lossFunc)

        # Updates best weights
        if costFunc < minCost:
            bestWeights = weights.copy()
            minCost = costFunc
            
        # Caches cost function every plotN iterations
        if it%plotN == 0:
            cost.append(costFunc)
            iteration.append(it)

        # "Backward" propagation (Newton-Raphson's Method) loop
        for i in range(numLayers+1, 0, -1):

            # Gets current layer weights
            Wi = np.copy(weights['W'+str(i)])
            bi = np.copy(weights['b'+str(i)])
            Ai = np.copy(Avals['A'+str(i)])

            # If on the first layer, APrev = X; else APrev = Ai-1
            if i == 1:
                APrev = np.copy(X)
            else:
                APrev = np.copy(Avals['A'+str(i-1)])

            # If on the last layer, dZi = Ai - Y; else dZi = (Wi+1 . dZi+1) * (Ai*(1-Ai))
            if i == numLayers+1:
                dZi = (Ai - Y)  # /(Ai * (1 - Ai)) ???
            else:
                dZi = np.dot(Wnxt.T, dZnxt) * Ai * (1 - Ai)

            # Calculates gradient vector (actually a matrix) of i-th layer
            m = APrev.shape[1]
            gradVecti = np.dot(APrev, dZi.T)/m
            gradbi = np.sum(dZi, axis = 1, keepdims = 1)/m
            gradVecti = np.append(gradVecti, gradbi.T, axis = 0)

            # Performs newton method on each node of i-th layer
            try:
                for j in range(len(Ai)):
                    
                    # Creates hessian matrix for node j in layer i
                    hessMatxi = np.dot(np.array([Ai[j]]), (1-np.array([Ai[j]])).T) * np.dot(APrev, APrev.T)/m
                    hessbipar = np.dot(np.array([Ai[j]]), (1-np.array([Ai[j]])).T) * np.dot(APrev, np.ones((APrev.shape[1],1)))/m
                    hessbi = np.dot(np.array([Ai[j]]), (1-np.array([Ai[j]])).T)/m
                    hessMatxi = np.concatenate((hessMatxi, hessbipar), axis = 1)
                    hessbipar = np.concatenate((hessbipar, hessbi), axis = 0)
                    hessMatxi = np.concatenate((hessMatxi, hessbipar.T), axis = 0)
                    hessMatxi = aprox_pos_def(hessMatxi)

                    # Creates descent direction for layer i
                    if j == 0:
                        deltai = np.linalg.solve(hessMatxi, np.array([gradVecti[:,j]]).T).T
                    else:
                        deltai = np.append(deltai, np.linalg.solve(hessMatxi, np.array([gradVecti[:,j]]).T).T, axis = 0)
            except:
                print("Singular matrix found when calculating descent direction; terminating computation.")
                break_code = 1
                break

            # Descent step for weights and biases
            dWi = deltai[:,:-1]
            dbi = np.array([deltai[:,-1]]).T

            # Cache dZi, Wi, bi
            dZnxt = np.copy(dZi)
            Wnxt = np.copy(Wi)     

            # Updates weights and biases
            Wi = Wi - learningRate*dWi
            bi = bi - learningRate*dbi
            weights['W'+str(i)] = Wi
            weights['b'+str(i)] = bi

        # Plot cost every plotN iterations
        if it % plotN == 0:
            plt.clf()
            plt.plot(iteration, cost, color = 'b')
            plt.xlabel("Iteration")
            plt.ylabel("Cost Function")
            plt.title("Newton-Raphson Descent for Cost Function")
            plt.pause(0.001)
            
        # End of backprop loop ==================================================

        # Early breaking condition met
        if break_code:
            AL = Avals['A'+str(numLayers+1)]
            lossFunc = -(Y*np.log(AL) + (1-Y)*np.log(1-AL))
            costFunc = np.mean(lossFunc)
            cost.append(costFunc)
            iteration.append(it)
            break

    # Plots cost function over time
    plt.clf()
    plt.plot(iteration, cost, color = 'b')
    plt.xlabel("Iteration")
    plt.ylabel("Cost Function")
    plt.title("Newton-Raphson Descent for Cost Function")
    plt.show(block = 0)

    return bestWeights

# Predicts if X vector tag is 1 or 0
def predict(weights, X, numLayers):

    A = forwardPropagation(weights, X, numLayers)
    A = A['A'+str(numLayers+1)]

    return (A > 0.5)

def example():

    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler as StdScaler
    from sklearn.model_selection import train_test_split

    # Won't work without scalling data
    scaler = StdScaler()
    data = load_breast_cancer(return_X_y=True)

    X = data[0]
    scaler.fit(X)
    X = scaler.transform(X)
    y = data[1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
    
    X_train = X_train.T
    X_test = X_test.T
    y_train = np.array([y_train])
    y_test = np.array([y_test])

    layers = [5, 5, 5]
    weights = model(X_train, y_train, layers, 0.1, max_iter = 500, plotN = 10)
    
    pred = predict(weights, X_train, len(layers))
    percnt = 0
    for i in range(pred.shape[1]):
        if pred[0,i] == y_train[0,i]:
            percnt += 1
    percnt /= pred.shape[1]
    print()
    print("Accuracy of", percnt*100 , "% on training set")

    pred = predict(weights, X_test, len(layers))
    percnt = 0
    for i in range(pred.shape[1]):
        if pred[0,i] == y_test[0,i]:
            percnt += 1
    percnt /= pred.shape[1]
    print()
    print("Accuracy of", percnt*100 , "% on test set")
    
example()








