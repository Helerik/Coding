#!/usr/bin/env python3
# Author: Erik Davino Vincent

import numpy as np
import matplotlib.pyplot as plt

# Fits for X and Y
def model(X, Y, layer_sizes, learning_rate, max_iter = 100, plot_N = 100):

    n_x, m_x = X.shape
    n_y, m_y = Y.shape
    if m_x != m_y:
        raise ValueError(f"Invalid vector sizes for X and Y -> X size = {X.shape} while Y size = {Y.shape}.")

    weights = initialize_weights(n_x, n_y, layer_sizes)
    num_layers = len(layer_sizes)

    return gradient_descent(X, Y, weights, learning_rate, num_layers, max_iter, plot_N)

# Sigmoid activation function
def sigmoid(t):
    return 1/(1+np.exp(-t))

# Initializes weights for each layer
def initialize_weights(n_x, n_y, layer_sizes, seed = None):
    
    np.random.seed(seed)
    
    weights = {}
    n_h_prev = n_x
    for i in range(len(layer_sizes)):
        n_h = layer_sizes[i]
        weights['W'+str(i+1)] = np.random.randn(n_h, n_h_prev)*np.sqrt(1/n_h_prev)
        weights['b'+str(i+1)] = np.zeros((n_h,1))
        n_h_prev = n_h
    
    weights['W'+str(len(layer_sizes)+1)] = np.random.random((n_y, n_h_prev))*np.sqrt(1/n_h_prev)
    weights['b'+str(len(layer_sizes)+1)] = np.zeros((n_y,1))

    np.random.seed(None)
    
    return weights

# Performs foward propagation
def forward_propagation(weights, X, num_layers):

    # Cache for A
    A_vals = {}
    Ai_prev = np.copy(X)
    for i in range(num_layers+1):
        
        Wi = weights['W'+str(i+1)]
        bi = weights['b'+str(i+1)]
        
        Zi = np.dot(Wi, Ai_prev) + bi
        Ai = sigmoid(Zi)

        A_vals['A'+str(i+1)] = Ai
        Ai_prev = np.copy(Ai)

    return A_vals
    

# Newton method for training a neural network
def gradient_descent(X, Y, weights, learning_rate, num_layers, max_iter, plot_N):
    
    # Cache for ploting cost
    cost = []
    iteration = []

    # Cache for minimum cost and best weights
    best_weights = weights.copy()
    min_cost = np.inf

    # Init break_code = 0
    break_code = 0
    for it in range(max_iter):

        # Forward propagation
        A_vals = forward_propagation(weights, X, num_layers)

        # Evaluates cost fucntion
        AL = np.copy(A_vals['A'+str(num_layers+1)])
        loss_func = -(Y*np.log(AL) + (1-Y)*np.log(1-AL))
        cost_func = np.mean(loss_func)

        # Updates best weights
        if cost_func < min_cost:
            best_weights = weights.copy()
            min_cost = cost_func
            
        # Caches cost function every plot_N iterations
        if it%plot_N == 0:
            cost.append(cost_func)
            iteration.append(it)

        # Backward propagation
        for i in range(num_layers+1, 0, -1):

            # Gets current layer weights
            Wi = np.copy(weights['W'+str(i)])
            bi = np.copy(weights['b'+str(i)])
            Ai = np.copy(A_vals['A'+str(i)])

            if i == 1:
                A_prev = np.copy(X)
            else:
                A_prev = np.copy(A_vals['A'+str(i-1)])

            # If on the last layer, dZi = Ai - Y; else dZi = (Wi+1 . dZi+1) * (Ai*(1-Ai))
            if i == num_layers+1:
                dZi = Ai - Y
            else:
                dZi = np.dot(Wnxt.T, dZnxt) * Ai * (1 - Ai)

            # Calculates dWi and dbi
            m = A_prev.shape[1]
            dWi = np.dot(A_prev, dZi.T)/m
            dbi = np.sum(dZi, axis = 1, keepdims = 1)/m

            # Cache dZi, Wi
            dZnxt = np.copy(dZi)
            Wnxt = np.copy(Wi)     

            # Updates weights and biases
            Wi = Wi - learning_rate*dWi.T
            bi = bi - learning_rate*dbi
            weights['W'+str(i)] = Wi
            weights['b'+str(i)] = bi

        # Plot cost every plot_N iterations
        if it % plot_N == 0:
            plt.clf()
            plt.plot(iteration, cost, color = 'b')
            plt.xlabel("Iteration")
            plt.ylabel("Cost Function")
            plt.title("Newton-Raphson Descent for Cost Function")
            plt.pause(0.001)
            
        # End of backprop loop ==================================================

        # Early breaking condition met
        if break_code:
            AL = A_vals['A'+str(num_layers+1)]
            loss_func = -(Y*np.log(AL) + (1-Y)*np.log(1-AL))
            cost_func = np.mean(loss_func)
            cost.append(cost_func)
            iteration.append(it)
            break

    # Plots cost function over time
    plt.clf()
    plt.plot(iteration, cost, color = 'b')
    plt.xlabel("Iteration")
    plt.ylabel("Cost Function")
    plt.title("Gradient Descent for Cost Function")
    plt.show(block = 0)

    return best_weights

# Predicts if X vector tag is 1 or 0
def predict(weights, X, num_layers):

    A = forward_propagation(weights, X, num_layers)
    A = A['A'+str(num_layers+1)]

    return A > 0.5

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    
    X_train = X_train.T
    X_test = X_test.T
    y_train = np.array([y_train])
    y_test = np.array([y_test])

    layers = [10,10]
    weights = model(X_train, y_train, layers, 0.1, max_iter = 500, plot_N = 10)
    
    pred = predict(weights, X_train, len(layers))
    percnt = 0
    for i in range(pred.shape[1]):
        if pred[0,i] == y_train[0,i]:
            percnt += 1
    percnt /= pred.shape[1]
    print()
    print(f"Accuracy of {percnt:.2%} on training set")

    pred = predict(weights, X_test, len(layers))
    percnt = 0
    for i in range(pred.shape[1]):
        if pred[0,i] == y_test[0,i]:
            percnt += 1
    percnt /= pred.shape[1]
    print()
    print(f"Accuracy of {percnt:.2%} on test set")
    
example()








